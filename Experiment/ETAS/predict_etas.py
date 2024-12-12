import json
import logging
import sys

from etas.evaluation import ETASLikelihoodCalculation
from etas import set_up_logger

set_up_logger(level=logging.DEBUG)

config = sys.argv[1]

if __name__ == '__main__':
    # reads configuration for example ETAS parameter inversion
    with open("config/"+ config+".json", 'r') as f:
        inversion_config = json.load(f)

    with open(inversion_config["data_path"]+"/parameters_0.json", 'r') as f:
        inversion_output = json.load(f)

    #这是一个class，继承了ETASparameterCalculation，这里就是直接通过_int_获取一下parameters.json中的参数
    calculation = ETASLikelihoodCalculation(inversion_output) 

    # prepare 准备数据，时间排序等 
    #获取Test时间段的地震数据 self.indexes_in_test_window
    #这里的n是用于_precompute_integral 中 simulate_aftershock_time 函数的参数size
    calculation.prepare(n=1000000)  

    calculation.evaluate_baseline_poisson_model() #这里计算的是泊松分布的三个指标
    nll, sll, tll = calculation.evaluate() #计算ETAS的三个指标，具体见evaluate函数的注释
    calculation.store_results(inversion_config['data_path']) #将数据存储到那个地方