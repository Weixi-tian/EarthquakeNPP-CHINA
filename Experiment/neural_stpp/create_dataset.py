import yaml, sys, os
import pandas as pd
from dotwiz import DotWiz
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import torch

with open(sys.argv[1], 'r') as file:
	config_dict = yaml.safe_load(file)

config = DotWiz(config_dict)

filepath = "data/earthquakes/"+config.data.init_args.name+".npz"

def find_largest_time_interval(df, target_event_count=3):
    # Ensure the DataFrame is sorted by time
    df = df.sort_values(by='time')
    
    # Initialize variables to store the maximum interval and corresponding number of events
    max_interval = pd.Timedelta(0)  # Start with the smallest possible interval (0)
    
    # Iterate over the DataFrame to find the largest interval containing the target number of events
    for i in range(len(df) - target_event_count + 1):
        start_time = df['time'].iloc[i]
        end_time = df['time'].iloc[i + target_event_count - 1]
        interval = end_time - start_time
        
        if interval > max_interval:
            max_interval = interval
    
    # Convert the maximum interval to float days
    max_interval_days = max_interval / pd.Timedelta(days=1)
    
    return max_interval_days, max_interval

def plot_event_histogram(df, interval, title):
	# Ensure the DataFrame is sorted by time
	df = df.sort_values(by='time')
	
	start_time = df['time'].min()
	end_time = start_time + interval
	event_counts = []
	
	while start_time <= df['time'].max():
		events_in_interval = df[(df['time'] >= start_time) & (df['time'] < end_time)]
		event_counts.append(len(events_in_interval))
		start_time = end_time
		end_time = start_time + interval
	
	plt.figure(figsize=(10, 6))
	plt.hist(event_counts, bins=150, edgecolor='black')
	plt.title(title)
	plt.xlabel('Number of Events')
	plt.ylabel('Frequency')
	plt.show()





# if not os.path.exists(filepath):
if True:

	df = pd.read_csv(
					config.catalog.path,
					parse_dates=["time"],
					dtype={"url": str, "alert": str},
				)
	df = df.sort_values(by='time')

	df = df[['time','x','y','magnitude']]


	### filter events by magnitude threshold

	df = df[df['magnitude']>=config.catalog.Mcut]

	optimal_T_days, optimal_T = find_largest_time_interval(df[df['time']>=config.catalog.train_nll_start])
	print(config.data.init_args.name)
	print(f"Biggest time interval containing 3 events: {optimal_T_days}")


	### create train/val/test dfs
	aux_df = df[df['time']>=config.catalog.auxiliary_start]
	aux_df = df[df['time']<config.catalog.train_nll_start]

	train_df = df[df['time']>=config.catalog.train_nll_start]
	train_df = train_df[train_df['time']< config.catalog.val_nll_start]

	val_df = df[df['time']>=config.catalog.val_nll_start]
	val_df = val_df[val_df['time']< config.catalog.test_nll_start]

	test_df = df[df['time']>=config.catalog.test_nll_start]
	test_df = test_df[test_df['time']< config.catalog.test_nll_end]


	# Plot histograms for train_df, val_df, and test_df
	# plot_event_histogram(train_df, optimal_T, 'Train Data Histogram')
	# plot_event_histogram(val_df, optimal_T, 'Validation Data Histogram')
	# plot_event_histogram(test_df, optimal_T, 'Test Data Histogram')


	## convert datetime to days

	train_df['time'] = (train_df['time']-train_df['time'].min()).dt.total_seconds() / (60*60*24)
	val_df['time'] = (val_df['time']-val_df['time'].min()).dt.total_seconds() / (60*60*24)
	test_df['time'] = (test_df['time']-test_df['time'].min()).dt.total_seconds() / (60*60*24)

	# List of DataFrames
	dfs = [train_df, val_df, test_df]

	# Process each DataFrame
	for i, df in enumerate(dfs):
		time_diffs = np.ediff1d(df['time'])

		# Identify the indices where the differences are less than or equal to 0
		indices_to_drop = np.where(time_diffs <= 0)[0] + 1

		indices_to_drop = df.index[indices_to_drop]

		# Drop the rows with the identified indices
		dfs[i] = df.drop(index=indices_to_drop)

	# Assign the processed DataFrames back
	train_df, val_df, test_df = dfs

	assert (np.ediff1d(train_df['time']) > 0).all()
	assert (np.ediff1d(val_df['time']) > 0).all()
	assert (np.ediff1d(test_df['time']) > 0).all()

	### drop magnitude column
	train_df.drop(columns=['magnitude'], inplace=True)
	val_df.drop(columns=['magnitude'], inplace=True)
	test_df.drop(columns=['magnitude'], inplace=True)


	### Format and store npz 

	train_ar = np.expand_dims(train_df.to_numpy(), axis=0)
	val_ar = np.expand_dims(val_df.to_numpy(), axis=0)
	test_ar = np.expand_dims(test_df.to_numpy(), axis=0)

	sequences = {'train':train_ar,'val':val_ar,'test':test_ar}
	np.savez(filepath, **sequences)


#######################################################################################


# Since NSTPP standardises the spatial region, we need to calculate \log(det(Sigma))
# to subtract from the final spatial log-likelihood score

	def standardize(dataset):
		dataset = [torch.tensor(seq) for seq in dataset]
		full = torch.cat(dataset, dim=0)
		S = full[:, 1:]
		S_mean = S.mean(0, keepdims=True)
		S_std = S.std(0, keepdims=True)
		return S_mean, S_std

	_, std = standardize(sequences['train'])

	log_det_inv = -np.log((std[0][0]*std[0][1]))
	print('log_det_inv: ',log_det_inv)
