# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import read_config
from layers.moe_layer.sub.moe_ADSNet import ADSNet_Model
from layers.moe_layer.sub.moe_LightNet import LightNet_Model
from layers.moe_layer.MOE import MOE_Model
from layers.moe_layer.MOE_noPre import MOENoPre_Model
from moe_main_generator import DataGenerator
import datetime
from moe_sub_scores import Model_eval
from utils import Plot_res
from moe_main_generator import getTimePeriod
# wjh辅助任务
from assitMoeModels import converAssit,converAssitByTime


def getHoursGridFromNPY_complete(filepath, delta_hour, ForecastHourNum):  # 20200619
    # m = config_dict['GridRowColNum']
    # n = config_dict['GridRowColNum']
    is_complete = True
    grid_list = []
    param_list = ['QICE_ave3', 'QSNOW_ave3', 'QGRAUP_ave3', 'W_max', 'RAINNC']
    # delta_hour -= 6
    for s in param_list:
        if not os.path.exists(os.path.join(filepath, '{}.npy'.format(s))):
            is_complete = False
    return is_complete


def time_data_iscomplete(datetime_peroid, WRFFileDir, ForecastHourNum,TruthFileDirGrid, TruthHistoryHourNum):
    datetime_peroid = datetime_peroid.rstrip('\n')
    datetime_peroid = datetime_peroid.rstrip('\r\n')
    if datetime_peroid == '':
        return False
    is_complete = True

    ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
    # read WRF
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6

    filepath = os.path.join(WRFFileDir, ft.date().strftime("%Y%m%d"), str(nchour))

    if not getHoursGridFromNPY_complete(filepath, delta_hour, ForecastHourNum):
        is_complete = False

    # read labels
    for hour_plus in range(ForecastHourNum):
        dt = ddt + datetime.timedelta(hours=hour_plus)
        tFilePath = TruthFileDirGrid + dt.strftime('%Y%m%d%H%M') + '.npy'
        if not os.path.exists(tFilePath):
            is_complete = False

    # read history observations
    for hour_plus in range(TruthHistoryHourNum):
        dt = ddt + datetime.timedelta(hours=hour_plus - TruthHistoryHourNum)
        tFilePath = TruthFileDirGrid + dt.strftime('%Y%m%d%H%M') + '.npy'
        if not os.path.exists(tFilePath):
            is_complete = False

    # 这一部分的作用是，获取我们 历史那段时间 也就是和预测时间相同长度时间 的obs和 wrf
    # 用来获取obs对于每个专家的注意力的权重 随着时间演化的趋势
    ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M') + datetime.timedelta(
        hours=config_dict['TruthHistoryHourNum'])
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6

    filepath = os.path.join(config_dict['WRFFileDir'], ft.date().strftime("%Y%m%d"), str(nchour))

    if not os.path.exists(filepath):
        is_complete = False

    return is_complete


def selectModel(config_dict):

    if config_dict['NetName'] == 'ADSNet':
        model = ADSNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'],
                             obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                  wrf_channels=config_dict['WRFChannelNum'],
                             row_col=config_dict['GridRowColNum']).to(config_dict['Device'])
    elif config_dict['NetName'] == 'LightNet':
        model = LightNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'],
                               obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'],
                               row_col=config_dict['GridRowColNum']).to(config_dict['Device'])
    elif config_dict['NetName'] == 'MOE':
        model = MOE_Model(truth_history_hour_num=config_dict['TruthHistoryHourNum'],
                forecast_hour_num=config_dict['ForecastHourNum'],
                row_col=config_dict['GridRowColNum'], wrf_channels=config_dict['WRFChannelNum'],obs_channel = 1).to(config_dict['Device'])
    elif config_dict['NetName'] == 'MOENoPre':
        model = MOENoPre_Model(truth_history_hour_num=config_dict['TruthHistoryHourNum'],
                forecast_hour_num=config_dict['ForecastHourNum'],
                row_col=config_dict['GridRowColNum'], wrf_channels=config_dict['WRFChannelNum'],obs_channel = 1).to(config_dict['Device'])
    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model


def DoTrain(config_dict):
    # model
    model = selectModel(config_dict)

    # data index
    TrainSetFilePath = 'data_index/TrainCase.txt'
    ValSetFilePath = 'data_index/ValCase.txt'
    TestSetFilePath = 'data_index/TestCase.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            line = line.rstrip('\n').rstrip('\r\n')
            # 由于数据不全 所以需要校验数据的完整
            if time_data_iscomplete(line, WRFFileDir=config_dict['WRFFileDir'],
                                    ForecastHourNum=config_dict['ForecastHourNum'],
                                    TruthFileDirGrid=config_dict['TruthFileDirGrid'],
                                    TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):
                train_list.append(line.rstrip('\n').rstrip('\r\n'))
    val_list = []
    with open(ValSetFilePath) as file:
        for line in file:
            line = line.rstrip('\n').rstrip('\r\n')

            # 由于数据不全 所以需要校验数据的完整
            if time_data_iscomplete(line, WRFFileDir=config_dict['WRFFileDir'],
                                    ForecastHourNum=config_dict['ForecastHourNum'],
                                    TruthFileDirGrid=config_dict['TruthFileDirGrid'],
                                    TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):
                val_list.append(line.rstrip('\n').rstrip('\r\n'))


    test_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            line = line.rstrip('\n').rstrip('\r\n')
            # 由于数据不全 所以需要校验数据的完整
            if time_data_iscomplete(line, WRFFileDir=config_dict['WRFFileDir'],
                                    ForecastHourNum=config_dict['ForecastHourNum'],
                                    TruthFileDirGrid=config_dict['TruthFileDirGrid'],
                                    TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):

                test_list.append(line.rstrip('\n').rstrip('\r\n'))



    print('一共有训练集合={},验证集={},测试集={}'.format(len(train_list), len(val_list), len(test_list)))

    # data
    train_data = DataGenerator(train_list, config_dict)
    train_loader = DataLoader(dataset=train_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)
    val_data = DataGenerator(val_list, config_dict)
    val_loader = DataLoader(dataset=val_data, batch_size=config_dict['Batchsize'], shuffle=False, num_workers=0)
    test_data = DataGenerator(test_list, config_dict)
    test_loader = DataLoader(dataset=test_data, batch_size=config_dict['Batchsize'], shuffle=False, num_workers=0)

    # model
    model = selectModel(config_dict)


    # loss function
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))

    # wjh test
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(25))


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])

    # eval
    model_eval_valdata = Model_eval(config_dict, is_save_model=False)
    model_eval_testdata = Model_eval(config_dict, is_save_model=True)

    # plot ETS
    ets_plot = Plot_res('./plot', ['val', 'test'], 'sumETS', 'epoch', 'sumETS', enable=True)

    if config_dict['NetName'] == 'MOE':
        print('Beginning train model = {} '.format(config_dict['NetName']))

        for epoch in range(config_dict['EpochNum']):

            for i, (X, y) in enumerate(train_loader):
                wrf, obs, wrf_old= X
                label = y
                wrf = wrf.to(config_dict['Device'])
                obs = obs.to(config_dict['Device'])
                wrf_old = wrf_old.to(config_dict['Device'])
                label = label.to(config_dict['Device'])

                if config_dict['NetName'] == 'MOE' or config_dict['NetName'].__contains__('MOE'):
                    pre_frames = model(wrf, obs, wrf_old)
                else :
                    pre_frames, h = model(wrf, obs)

                # wjh辅助任务
                #p_assit,l_assit = converAssitByTime(pre_frames,label)
                #loss_assit = criterion(torch.flatten(p_assit), torch.flatten(l_assit))
                # backward
                optimizer.zero_grad()

                # loss = criterion(torch.flatten(pre_frames), torch.flatten(label)) + loss_assit
                # print('开启了辅助任务 + loss_assit Bytime')

                loss = criterion(torch.flatten(pre_frames), torch.flatten(label))
                print('关闭了辅助任务 ')
                loss = loss


                loss.backward()

                # update weights
                optimizer.step()

                # output
                print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i + 1, len(train_loader), loss.item()))
                # pod, far, ts, ets = train_calparams_epoch.cal_batch(label, pre_frames)
                # sumpod, sumfar, sumts, sumets = train_calparams_epoch.cal_batch_sum(label, pre_frames)
                # info = 'TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}\nPOD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}\nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n'\
                #     .format(epoch, i+1, len(train_loader), loss.item(), pod, far, ts, ets, sumpod, sumfar, sumts, sumets)
                # print(info)
            val_sumets = model_eval_valdata.eval(val_loader, model, epoch)
            test_sumets = model_eval_testdata.eval(test_loader, model, epoch)
            ets_plot.step([val_sumets, test_sumets])
    else :
        print('Beginning train model = {}'.format(config_dict['NetName']))
        for epoch in range(config_dict['EpochNum']):
            # train_calparams_epoch = Cal_params_epoch()
            for i, (X, y) in enumerate(train_loader):
                wrf, obs, wrf_old = X
                label = y
                wrf = wrf.to(config_dict['Device'])
                obs = obs.to(config_dict['Device'])
                wrf_old = wrf_old.to(config_dict['Device'])

                label = label.to(config_dict['Device'])

                if config_dict['NetName'] == 'MOE' or config_dict['NetName'].cont:
                    pre_frames, h = model(wrf, obs,wrf_old)
                else:
                    pre_frames, h = model(wrf,obs)

                # wjh辅助任务 总和
                p_assit,l_assit = converAssit(pre_frames,label)
                loss1 = criterion(torch.flatten(p_assit), torch.flatten(l_assit))

                # backward
                optimizer.zero_grad()
                loss = criterion(torch.flatten(pre_frames), torch.flatten(label)) + loss1

                loss.backward()

                # update weights
                optimizer.step()

                # output
                print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i + 1, len(train_loader), loss.item()))
                # pod, far, ts, ets = train_calparams_epoch.cal_batch(label, pre_frames)
                # sumpod, sumfar, sumts, sumets = train_calparams_epoch.cal_batch_sum(label, pre_frames)
                # info = 'TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}\nPOD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}\nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n'\
                #     .format(epoch, i+1, len(train_loader), loss.item(), pod, far, ts, ets, sumpod, sumfar, sumts, sumets)
                # print(info)
            val_sumets = model_eval_valdata.eval(val_loader, model, epoch)
            test_sumets = model_eval_testdata.eval(test_loader, model, epoch)
            ets_plot.step([val_sumets, test_sumets])


    # SelectEpoch(modelrecordname)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    config_dict = read_config()

    print('开始训练，参数为={}'.format(config_dict))

    if not os.path.isdir(config_dict['ModelFileDir']):
        os.makedirs(config_dict['ModelFileDir'])

    if not os.path.isdir(config_dict['RecordFileDir']):
        os.makedirs(config_dict['RecordFileDir'])

    if not os.path.isdir(config_dict['VisResultFileDir']):
        os.makedirs(config_dict['VisResultFileDir'])

    # train
    DoTrain(config_dict)



