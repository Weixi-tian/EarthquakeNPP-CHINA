import json
import logging
import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

import sys

sys.path.insert(1, '../etas/')

# from __init__ import set_up_logger
from etas.inversion import round_half_up
from etas.simulation import generate_catalog

# set_up_logger(level=logging.INFO)
# if __name__ == '__main__':
def run_simulation():
    # reads configuration for example ETAS parameter inversion
    with open("/home/tianweixi/EarthquakeNPP_CSEP_China/Datasets/ETAS/simulate_ETAS_California_catalog_config.json", 'r') as f:
        simulation_config = json.load(f)

    print(simulation_config)

    polygon_coords=np.load(simulation_config["shape_coords"])
    polygon_coords = polygon_coords[0:12962, [1, 0]] #点文件就是有问题，若是用全部的点文件会导致很久都跑不出来
    region = Polygon(polygon_coords)

    # np.random.seed(777)

    #调用etas的generate_catalog 生成模拟目录（需要输入parameters等参数）
    synthetic = generate_catalog(
        polygon=region,
        timewindow_start=pd.to_datetime(simulation_config["burn_start"]),
        timewindow_end=pd.to_datetime(simulation_config["end"]),
        parameters=simulation_config["parameters"],
        mc=simulation_config["mc"],
        beta_main=simulation_config["beta"],
        delta_m=simulation_config["delta_m"]
    )

    synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    synthetic.index.name = 'id'
    print("store catalog..")
    primary_start = simulation_config['primary_start']
    fn_store = simulation_config['fn_store']
    os.makedirs(os.path.dirname(fn_store), exist_ok=True)
    synthetic[["latitude", "longitude", "time", "magnitude"]].query(
        "time>=@primary_start").to_csv(fn_store)
    print("\nDONE!")

    synthetic = synthetic.sort_values(by='time')
    plt.plot(synthetic['time'],range(len(synthetic['time'])))
    plt.show()

    plt.scatter(synthetic['time'],synthetic['magnitude'])
    plt.show()


if __name__ == '__main__':
    run_simulation()