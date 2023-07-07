import pandas as pd
import numpy as np


def labels(url):
	"""
	This function labels the stock data based on conditions evaluated on closing prices and rolling averages.

	The labels are assigned based on these conditions:
	1. The closing price is less than the 20-day and 50-day moving averages and the 50-day rolling standard deviation is greater than its median.
	2. The closing price is greater than the 20-day moving average, less than the 50-day moving average, and the 50-day rolling standard deviation is greater than its median.
	3. The closing price is greater than both the 20-day and 50-day moving averages and the 50-day rolling standard deviation is greater than its median.
	4. The closing price is greater than both the 20-day and 50-day moving averages and the 50-day rolling standard deviation is less than its median.
	5. The closing price is less than the 20-day moving average, greater than the 50-day moving average, and the 50-day rolling standard deviation is less than its median.
	6. The closing price is greater than the 20-day moving average, less than the 50-day moving average, and the 50-day rolling standard deviation is less than its median.

	:param url: The URL of the CSV file containing the stock data.
	:type url: str
	:return: The stock data DataFrame with an additional 'Label' column indicating the assigned condition.
	:rtype: pandas.DataFrame
	"""
	stock = pd.read_csv(url)
	rolling_20_mean = stock.Close.rolling(window=20).mean()[50:]
	rolling_50_mean = stock.Close.rolling(window=50).mean()[50:]
	rolling_50_std = stock.Close.rolling(window=50).std()[50:]
	stock = stock.iloc[50:, :]
	cond_0 = np.where((stock["Close"] < rolling_20_mean) & (stock["Close"] < rolling_50_mean) & (
				rolling_50_std > rolling_50_std.median()), 1, 0)
	cond_1 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] < rolling_50_mean) & (
				rolling_50_std > rolling_50_std.median()), 2, 0)
	cond_2 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] > rolling_50_mean) & (
				rolling_50_std > rolling_50_std.median()), 3, 0)
	cond_3 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] > rolling_50_mean) & (
				rolling_50_std < rolling_50_std.median()), 4, 0)
	cond_4 = np.where((stock["Close"] < rolling_20_mean) & (stock["Close"] > rolling_50_mean) & (
				rolling_50_std < rolling_50_std.median()), 5, 0)
	cond_5 = np.where((stock["Close"] > rolling_20_mean) & (stock["Close"] < rolling_50_mean) & (
				rolling_50_std < rolling_50_std.median()), 6, 0)

	cond_all = cond_0 + cond_1 + cond_2 + cond_3 + cond_4 + cond_5
	stock["Label"] = cond_all

	return stock


def end_cond(X_train):
	"""
	This function categorizes the data in X_train based on the percentage change from the initial value to the final value.

	The conditions for categorization are as follows:
	1. The percentage change is less than -10%.
	2. The percentage change is between -10% and -5%.
	3. The percentage change is between -5% and 0%.
	4. The percentage change is between 0% and 5%.
	5. The percentage change is between 5% and 10%.
	6. The percentage change is greater than 10%.

	The categories are represented by labels from 0 to 5 respectively. The categorized labels are then added as a new layer to the input tensor.

	:param X_train: The 3D input data tensor. The first dimension is the number of instances, the second dimension is the time steps, and the third dimension is the features of the data.
	:type X_train: numpy.ndarray
	:return: The input data tensor with an additional layer of categorized labels based on the percentage change.
	:rtype: numpy.ndarray
	"""
	vals = X_train[:, 23, 4] / X_train[:, 0, 4] - 1

	comb1 = np.where(vals < -.1, 0, 0)
	comb2 = np.where((vals >= -.1) & (vals <= -.05), 1, 0)
	comb3 = np.where((vals >= -.05) & (vals <= -.0), 2, 0)
	comb4 = np.where((vals > 0) & (vals <= 0.05), 3, 0)
	comb5 = np.where((vals > 0.05) & (vals <= 0.1), 4, 0)
	comb6 = np.where(vals > 0.1, 5, 0)
	cond_all = comb1 + comb2 + comb3 + comb4 + comb5 + comb6

	print(np.unique(cond_all, return_counts=True))
	arr = np.repeat(cond_all, 24, axis=0).reshape(len(cond_all), 24)
	X_train = np.dstack((X_train, arr))
	return X_train
