import pandas as pd
import numpy as np
import urllib.request
import requests
import tarfile
import shutil
import re
import os
import zipfile
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
from scipy.ndimage import uniform_filter1d
import geopandas as gpd


import time
# import cchardet 
import json
import cartopy.crs as ccrs
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



def load_CN_borders(path):
	with open(path) as CN_borders:
		context_borders = CN_borders.read()
		blocks_borders = [cnt for cnt in context_borders.split('>') if len(cnt.split()) > 0]
		#遍历每个 block，只处理包含数值的部分
		borders = []
		for block_border in blocks_borders:
			#去掉 block 中的注释行，只保留数值行
			clean_block_border = "\n".join([line for line in block_border.splitlines() if line.strip() and not line.strip().startswith("#")])
			#只有当 clean_block 中有内容时才解析为数组
			if clean_block_border:
				borders.append(np.fromstring(clean_block_border, dtype=float, sep=' ')) #fromstring 表示将每个block转换为由数值(浮点数)组成的numpy数组
	return borders


def load_CN_faluts(path):
	with open(path) as CN_faults:
		context_faults = CN_faults.read()
		blocks_faults = [cnt for cnt in context_faults.split('>') if len(cnt.split()) > 0]
		faluts = []
		for block_falut in blocks_faults:
			clean_block_fault = "\n".join([line for line in block_falut.splitlines() if line.strip() and not line.strip().startswith("#")])
			if clean_block_fault:
				faluts.append(np.fromstring(clean_block_fault, dtype=float, sep=' ')) #fromstring 表示将每个block转换为由数值(浮点数)组成的numpy数组
	return faluts


def plot_faluts(faluts,ax,min_lon,max_lon,min_lat,max_lat):
	for point in faluts:
		# 确保数据长度为偶数，且非空
		if len(point) > 1 and len(point) % 2 == 0:
			# 获取经纬度点对
			lons, lats = point[0::2], point[1::2]
			# 筛选出符合经纬度范围的数据
			filtered_lons = []
			filtered_lats = []
			for lon, lat in zip(lons, lats):
				if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
					filtered_lons.append(lon)
					filtered_lats.append(lat)
					# 只有当筛选后有有效点对时才绘制
					if len(filtered_lons) > 1:
						ax.plot(point[0::2], point[1::2], '-', color='#5C4033', transform=ccrs.PlateCarree(),zorder=10,linewidth=0.3,alpha=0.5)
					else:
						continue
					    # print(f"Skipping invalid point with length: {len(point)}")


#Plot borders/绘制国界、省界、十段线、海南诸岛数据
def plot_borders(borders,ax):
	for line in borders:
		# 确保数据长度为偶数，且非空
		if len(line) > 1 and len(line) % 2 == 0:
			ax.plot(line[0::2], line[1::2], '-', color='#2F2F2F', transform=ccrs.PlateCarree(),zorder=100,linewidth=0.7,alpha=0.6) #zorder 控制画图顺序,越大画上去越晚
		else:
			continue #有一些错误的数据点不是偶数成对,所以这里直接跳过,或者用下面的检验跳过了多少
		# print(f"Skipping invalid line with length: {len(line)}")

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def new_azimuthal_equidistant_projection(lat, lon, lat0, lon0, R=6371):
    """
    Forward azimuthal equidistant projection.
    Converts (lat, lon) to (x, y) with respect to a center (lat0, lon0).
    Parameters:
    - lat, lon: Coordinates to project (degrees)
    - lat0, lon0: Center of the projection (degrees)
    - R: Radius of the sphere (default: Earth radius in km)
    Returns:
    - x, y: Projected coordinates (km)
    """
    lat, lon, lat0, lon0 = map(deg2rad, [lat, lon, lat0, lon0])
    delta_lambda = lon - lon0
    c = np.arccos(np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(delta_lambda))
    k = np.where(c == 0, 0, R * c / np.sin(c))
    y = k * np.cos(lat) * np.sin(delta_lambda)
    x = k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(delta_lambda))
    return x, y

def azimuthal_equidistant_inverse(x, y, lat0, lon0, R=6371):
    """
    Inverse azimuthal equidistant projection.
    Converts (x, y) back to (lat, lon) from a center (lat0, lon0).
    Parameters:
    - x, y: Projected coordinates (km)
    - lat0, lon0: Center of the projection (degrees)
    - R: Radius of the sphere (default: Earth radius in km)
    Returns:
    - lat, lon: Geographic coordinates (degrees)
    """
    lat0, lon0 = deg2rad(lat0), deg2rad(lon0)
    r = np.sqrt(x**2 + y**2)
    c = r / R
    lat = np.where(r == 0, lat0, np.arcsin(np.cos(c) * np.sin(lat0) + (x * np.sin(c) * np.cos(lat0) / r)))
    lon = lon0 + np.arctan2(y * np.sin(c), r * np.cos(lat0) * np.cos(c) - x * np.sin(lat0) * np.sin(c))
    return rad2deg(lat), rad2deg(lon)

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# 修改后的提取函数，增加类型检查
def extract_parameters(data, parameter):
    """
    从 JSON 数据中提取指定参数的值，并将它们合并为一个列表。
    
    参数:
        data (list): 包含字典的列表，每个字典可能包含特定的参数键。
        parameter (str): 要提取的参数键名称。
    
    返回:
        list: 合并的指定参数数据列表。
    """
    output = []
    for results in data:
        if parameter in results:
            value = results[parameter]
            if isinstance(value, (list, tuple)):  # 检查是否为可迭代对象
                output.extend(value)
            else:
                output.append(value)  # 直接将非迭代对象添加到列表
    return output

def combined_parameters_and_catalog(parameters,catalog,length):
    """
Input:
    parameters : list of dict
        - `parameters` 是一个包含多个字典的列表，每个字典表示一次模拟的 ETAS 模型参数

    catalog : pandas.DataFrame
        - `catalog` is Pandas DataFrame

    length : int
        - `length` represents the length of NaN column

Output:
    pandas.DataFrame"
    "Return a df including catalog with their parameters information
    """
    # 查看参数名
    print('Parameters name are:', parameters[0].keys())
    all_parameters_df = pd.DataFrame()  # 用于存储所有参数的 DataFrame

    for key in parameters[0].keys():
        output = extract_parameters(parameters, key)

        # 处理二维数组 (N, M)
        if isinstance(output, list) and isinstance(output[0], (list, np.ndarray)):
            output = np.array(output)  # 转为 NumPy 数组
            # 创建二维 NaN 填充 (20, 列数)，每个元素是一个长度为 M 的 NaN 列表
            nan_padding = [[np.nan] * output.shape[1]] * length
            output_padded = nan_padding + output.tolist()  # 拼接 NaN 和原始数据
            # 为每个参数创建 DataFrame，每一行是一个列表
            output_df = pd.DataFrame({key: output_padded})

            # 合并到总的参数 DataFrame 中
            all_parameters_df = pd.concat([all_parameters_df, output_df], axis=1)
    return pd.concat([catalog, all_parameters_df],axis=1)





#rescale values from normalization to original coordinate
def rescale_from_normalization_to_actual_distance(normazation_value,scales,bias):
    rescale_value = normazation_value * scales + bias
    return rescale_value

#Gaussian pdf calculation(# 定义一维高斯分布的概率密度函数 (PDF))
def gaussian_1d_pdf(distances_km,sigma):
    return np.exp(-distances_km**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

#Gaussian pdf calculation(# 定义二维高斯分布的概率密度函数 (PDF))
def gaussian_2d_pdf(x, y, mu, sigma_x, sigma_y, rho=0):
    """计算二维高斯分布的概率密度值"""
    x_mu = x - mu[0]
    y_mu = y - mu[1]
    
    z = (1.0 / (2.0 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))) * \
        np.exp(- (1.0 / (2 * (1 - rho**2))) * 
               ( (x_mu**2 / sigma_x**2) + 
                 (y_mu**2 / sigma_y**2) - 
                 (2 * rho * x_mu * y_mu) / (sigma_x * sigma_y) ))
    return z

def simulate_aftershock_place(log10_d, gamma, rho, mi, mc,sample_point =300000):
    # x and y offset in km
    d = np.power(10, log10_d)
    d_g = d * np.exp(gamma * (mi - mc))
    y_r = np.random.uniform(size=sample_point)
    r = np.sqrt(np.power(1 - y_r, -1 / rho) * d_g - d_g)
    phi = np.random.uniform(0, 2 * np.pi, size=sample_point)

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    return x, y


#  **定义地震空间核函数**
def space_decay(spatial_distance_squared, mc, m, d, gamma, rho):
	d = np.power(10,d)
	return 1 / np.power((spatial_distance_squared + d * np.exp(gamma * (m - mc))), (1 + rho))

def hav(theta):
    return np.square(np.sin(theta / 2))

def haversine(lat_rad_1, lat_rad_2, lon_rad_1, lon_rad_2, earth_radius=6.3781e3):
    """
    Calculates the distance on a sphere.
    """
    d = (
        2
        * earth_radius
        * np.arcsin(
            np.sqrt(
                hav(lat_rad_1 - lat_rad_2)
                + np.cos(lat_rad_1) * np.cos(lat_rad_2)
                * hav(lon_rad_1 - lon_rad_2)
            )
        )
    )
    return d

def compute_dist_squared_from_history_events(history_events, lat_rads: np.ndarray, long_rads: np.ndarray, earth_radius=6.3781e3):
    """
    Computes squared distance between historical earthquake event and all grid points.
    """
    lat_ref = np.full(lat_rads.shape, history_events[0])  # 确保形状一致
    lon_ref = np.full(long_rads.shape, history_events[1])  # 确保形状一致
    return np.square(haversine(lat_ref, lat_rads, lon_ref, long_rads, earth_radius))


