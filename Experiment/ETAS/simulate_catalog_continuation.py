#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# simulation of catalog continuation (for forecasting)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
###############################################################################


import json
import logging, sys

from etas import set_up_logger
sys.path.append('etas/')
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation
from datetime import datetime, timedelta

set_up_logger(level=logging.INFO)

if __name__ == '__main__':
    # read configuration in
    # '../config/[dataset].json'
    # this should contain the path to the parameters_*.json file
    # that is produced when running invert_etas.py.

    # read script arguments
    with open('config/'+sys.argv[1]+'.json', 'r') as f:
        config = json.load(f)

    days_from_start = int(sys.argv[2])
    
    #这部分一直到inversion_Output自己定义了和原始不一样的参数，包括时间，路径等
    fn_inversion_output = config['data_path']+'/parameters_0.json'
    # fn_store_simulation = config['data_path']+'/day_'+days_from_start+'.csv'
    fn_store_simulation = config['data_path']+'/day_'+sys.argv[1]+'.csv'
    forecast_duration = 1

    # load output from inversion
    with open(fn_inversion_output, 'r') as f:
        inversion_output = json.load(f)
    
    inversion_output['three_dim']=False
    inversion_output['space_unit_in_meters']=False #关掉，因为前面的invert结果中没有这两个参数

    # Set day of forecast
    inversion_output['timewindow_end'] = (datetime.strptime(inversion_output['timewindow_end'], '%Y-%m-%d %H:%M:%S')+timedelta(days=days_from_start)).strftime('%Y-%m-%d %H:%M:%S')
    

    etas_inversion_reload = ETASParameterCalculation.load_calculation(
        inversion_output) #就是用load_calculation函数加载 invert输出的数据，前面关掉了一些参数

    # initialize simulation
    simulation = ETASSimulation(etas_inversion_reload) #读取前面获取的参数，实际上也就是被改变的invert输出的那一组东西
    simulation.prepare() #一些简单的计算，主要是做一些准备工作，获取数据啥的

    # simulate and store one catalog
    simulation.simulate_to_csv(fn_store_simulation, forecast_duration, n_simulations = 10000,i_start=0)
