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


def download_USGS_data(start_year, start_month,end_year,end_month, max_lat, min_lat, max_lon, min_lon, minimum_magnitude):
	start_time = datetime(start_year,start_month,1) #转换时间
	end_time = datetime(end_year,end_month,1)
	url = (
		"https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
		f"starttime={start_time.strftime('%Y-%m-%dT%H:%M:%S')}&endtime={end_time.strftime('%Y-%m-%dT%H:%M:%S')}"
		f"&maxlatitude={max_lat}&minlatitude={min_lat}&maxlongitude={max_lon}"
		f"&minlongitude={min_lon}&minmagnitude={minimum_magnitude}&eventtype=earthquake&orderby=time-asc"
	)

	filename = f"原始目录/{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
	os.makedirs("原始目录", exist_ok=True)
	urllib.request.urlretrieve(url, filename)
	return filename



def download_China_data_CENC(start_year, start_month,start_day,end_year, end_month,end_day,max_lat, min_lat, max_lon, min_lon,minimum_magnitude):
	start_time = datetime(start_year,start_month,start_day) #转换时间
	end_time = datetime(end_year,end_month,end_day)
	#转换时间戳
	times = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	timeArray = time.strptime(times, '%Y-%m-%d %H:%M:%S')
	timeStamp= int(time.mktime(timeArray))
	time.sleep(5)
	
	#提供URL链接
	url = (
		"https://www.ceic.ac.cn/ajax/search?"
		f"page=1&&start={start_time.strftime('%Y-%m-%d %H:%M:%S')}&&end={end_time.strftime('%Y-%m-%d %H:%M:%S')}"
		f"&&jingdu1={min_lon}&&jingdu2={max_lon}&&weidu1={min_lat}&&weidu2={max_lat}&&height1=&&height2="
		f"&&zhenji1={minimum_magnitude}&&zhenji2=&&callback=jQuery180017662835951203926_1727276133223&_={timeStamp}"
	)
	
	#请求抓取
	response = requests.get(url, verify=False)  # 忽略 SSL 验证
	
	#转码解码
	encoding=cchardet.detect(response.content)['encoding']
	html=response.content.decode(encoding)
	# print(html)
	
	#删除返回的 jQuery 回调函数前缀，因为该部分无法解码
	json_data = re.sub(r'jQuery\d+_\d+\(', '', html)  # 匹配并删除前面的回调前缀
	json_data = json_data[:-1]  # 删除最后一个括号
	# print(json_data)
	
	# 获取总页数
	# total_pages = int(json_data[-2])
	total_pages = int(json_data.rsplit(':', 1)[-1].rstrip('}'))
	print("pages is :", total_pages)
	
	# #转换成字典
	# json_data = eval(json_data)
	
	filenames = []
	# 循环抓取每一页的数据
	for page in range(1, total_pages + 1):
		url = (
			f"https://www.ceic.ac.cn/ajax/search?"
			f"page={page}&&start={start_time.strftime('%Y-%m-%d %H:%M:%S')}&&end={end_time.strftime('%Y-%m-%d %H:%M:%S')}"
			f"&&jingdu1={min_lon}&&jingdu2={max_lon}&&weidu1={min_lat}&&weidu2={max_lat}&&height1=&&height2="
			f"&&zhenji1={minimum_magnitude}&&zhenji2=&&callback=jQuery180017662835951203926_1727276133223&_={timeStamp}"
			)
		
		response = requests.get(url, verify=False)  # 忽略 SSL 验证
		
		#转码解码
		encoding=cchardet.detect(response.content)['encoding']
		html=response.content.decode(encoding)
		
		#删除返回的 jQuery 回调函数前缀，因为该部分无法解码
		json_data = re.sub(r'jQuery\d+_\d+\(', '', html)  # 匹配并删除前面的回调前缀
		json_data = json_data[:-1]  # 删除最后一个括号
		
		#转换成字典
		json_data = json.loads(json_data)
		
		all_data = []
		for i in json_data['shuju']:
			all_data.append([
				i['O_TIME'],  # 时间
				i['EPI_LAT'],  # 纬度
				i['EPI_LON'],  # 经度
				i['M'],  # 震级
				i['M_MS'],
				i['EPI_DEPTH'],  # 深度
				i['LOCATION_C'],  # 参考位置
				])
			
		df = pd.DataFrame(all_data, columns=['time', 'latitude', 'longitude', 'magnitude', 'Ms', 'depth', 'location'])
		
		os.makedirs("raw", exist_ok=True)
		
		filename = f"raw/page{page}.csv"
		
		df.to_csv(filename, index=False, encoding='utf-8')
		print(f"Data saved to {filename}")
		filenames.append(filename)
	return filenames
		


def download_China_data_github():
	url = (
		"https://raw.githubusercontent.com/Weixi-tian/CENC_Earthquake_Catalog/refs/heads/main/Dataset/20240509_china3ms.csv"
	)

	filenames = f"raw/20240509_china3ms.csv"
	os.makedirs("raw", exist_ok=True)
	urllib.request.urlretrieve(url, filenames)
	return filenames



def combine_csv_files(filenames, destination_path, chunksize=100000):
	with open(destination_path, 'w') as wfd:
		for i, f in enumerate(filenames):
			for chunk in pd.read_csv(f, chunksize=chunksize):
				chunk.to_csv(wfd, header=(i == 0), index=False)




def fmd(mag, mbin):

	mag = np.array(mag)
    #计算从小到大的震级区间，如（0.00， 0.05， 0.10， ... 3.00）
	mi = np.arange(min(np.round(mag/mbin)*mbin), max(np.round(mag/mbin)*mbin),mbin)

	nbm = len(mi)
	cumnbmag = np.zeros(nbm)
	nbmag = np.zeros(nbm)

	for i in range(nbm):
		cumnbmag[i] = sum((mag > mi[i]-mbin/2)) #依次计算累计地震数目

    #依次计算 每个时间区间i内地震震级 ＞ mi[i]-mbin/2 的数量的差异，比如第一个区间20个，第二个区间18个，差距nbmag=2
	#也就是累计地震数目的差异
	#因此这里nbmag表示每个时间区间的实际的地震数目个数
	cumnbmagtmp = np.append(cumnbmag,0)
	nbmag = abs(np.ediff1d(cumnbmagtmp))

	res = {'m':mi, 'cum':cumnbmag, 'noncum':nbmag}

	return res



def maxc(mag, mbin):

	FMD = fmd(mag, mbin)

	if len(FMD['noncum'])>0: #如果nbmag>0
	# if True:

		Mc = FMD['m'][np.where(FMD['noncum']==max(FMD['noncum']))[0]][0]

	else:
		Mc = None

	return Mc


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
						ax.plot(point[0::2], point[1::2], '-', color='gray', transform=ccrs.PlateCarree(),zorder=10,linewidth=0.4)
					else:
						continue
					    # print(f"Skipping invalid point with length: {len(point)}")

#Plot borders/绘制国界、省界、十段线、海南诸岛数据

def plot_borders(borders,ax):
	for line in borders:
		# 确保数据长度为偶数，且非空
		if len(line) > 1 and len(line) % 2 == 0:
			ax.plot(line[0::2], line[1::2], '-', color='black', transform=ccrs.PlateCarree(),zorder=100,linewidth=0.7) #zorder 控制画图顺序,越大画上去越晚
		else:
			continue #有一些错误的数据点不是偶数成对,所以这里直接跳过,或者用下面的检验跳过了多少
		# print(f"Skipping invalid line with length: {len(line)}")
		


def azimuthal_equidistant_projection(latitude, longitude, center_latitude, center_longitude):
	R = 6371  # Earth's radius in kilometers
	phi1 = np.radians(center_latitude)
	phi2 = np.radians(latitude)
	delta_lambda = np.radians(longitude - center_longitude)

	delta_sigma = np.arccos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda))
	azimuth = np.arctan2(np.sin(delta_lambda), np.cos(phi1) * np.tan(phi2) - np.sin(phi1) * np.cos(delta_lambda))

	x = R * delta_sigma * np.cos(azimuth)
	y = R * delta_sigma * np.sin(azimuth)
	return x, y