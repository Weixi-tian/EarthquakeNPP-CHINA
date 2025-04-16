import yaml, sys, os
import pandas as pd
from dotwiz import DotWiz
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


with open(sys.argv[1], 'r') as file:
	config_dict = yaml.safe_load(file)

config = DotWiz(config_dict)


filepath = "data/spatiotemporal/"+config.data.init_args.name+".npz"

if not os.path.exists(filepath):
# if True:

	df = pd.read_csv(
	                config.catalog.path,
	                # index_col=0,
	                parse_dates=["time"],
	                dtype={"url": str, "alert": str},
	            )

	df = df.sort_values(by='time')

	df = df[['time','x','y','magnitude']]

	df = df[df['magnitude']>=config.catalog.Mcut]

	### create train/val/test dfs
	aux_df = df[df['time']>=config.catalog.auxiliary_start]
	aux_df = df[df['time']<config.catalog.train_nll_start]

	train_df = df[df['time']>=config.catalog.train_nll_start]
	train_df = train_df[train_df['time']< config.catalog.val_nll_start]

	val_df = df[df['time']>=config.catalog.val_nll_start]
	val_df = val_df[val_df['time']< config.catalog.test_nll_start]

	test_df = df[df['time']>=config.catalog.test_nll_start]
	test_df = test_df[test_df['time']< config.catalog.test_nll_end]


	### add burn-in events for sliding window

	lookback = 20

	train_df = pd.concat([aux_df.tail(lookback), train_df], ignore_index=True)
	val_df = pd.concat([train_df.tail(lookback), val_df], ignore_index=True)
	test_df = pd.concat([val_df.tail(lookback), test_df], ignore_index=True)


	## convert datetime to days

	train_df['time'] = (train_df['time']-train_df['time'].min()).dt.total_seconds() / (60*60*24)
	val_df['time'] = (val_df['time']-val_df['time'].min()).dt.total_seconds() / (60*60*24)
	test_df['time'] = (test_df['time']-test_df['time'].min()).dt.total_seconds() / (60*60*24)

	assert (np.ediff1d(train_df['time']) >= 0).all()
	assert (np.ediff1d(val_df['time']) >= 0).all()
	assert (np.ediff1d(test_df['time']) >= 0).all()
	
	# to judge whether save the magnitude to the dataset ====add by tianweixi
	if config.model.init_args.magnitude_information == False:
		### drop magnitude column
		train_df.drop(columns=['magnitude'], inplace=True)
		val_df.drop(columns=['magnitude'], inplace=True)
		test_df.drop(columns=['magnitude'], inplace=True)
	else:
		print('Dataset including magnitude at the last column')
		
	### Format and store npz 
	train_ar = np.expand_dims(train_df.to_numpy(), axis=0)
	val_ar = np.expand_dims(val_df.to_numpy(), axis=0)
	test_ar = np.expand_dims(test_df.to_numpy(), axis=0)
	
	sequences = {'train':train_ar,'val':val_ar,'test':test_ar}
	np.savez(filepath, **sequences)





