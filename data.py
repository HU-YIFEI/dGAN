import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def MinMax(train_data):
	"""
	This function applies MinMax scaling to the training data. The MinMax scaling is fit on the training data and
	then applied to it. The scaled data is then reshaped back into its original 3D shape.

	:param train_data: 3D numpy array containing the training data. Shape of array is (number of instances, number of time steps, number of features).
	:type train_data:
	:return: A tuple containing the scaled training data and the fitted scaler.
	:rtype: tuple(numpy.ndarray, sklearn.preprocessing.MinMaxScaler)
	"""
	scaler = MinMaxScaler()
	num_instances, num_time_steps, num_features = train_data.shape
	train_data = np.reshape(train_data, (-1, num_features)) # todo: what is the type of train_data?
	train_data = scaler.fit_transform(train_data)
	train_data = np.reshape(train_data, (num_instances, num_time_steps, num_features))
	return train_data, scaler


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

	#todo: examine feature engineering
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
	vals = X_train[:, 23, 1] / X_train[:, 0, 1] - 1

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


def data_loading(seq_length, data):
	"""
	This function loads stocks data from a csv file, applies some transformations to the data,
	and returns the transformed data along with the original data and a MinMaxScaler object.

	:param seq_length: The length of sequences for which the input data is to be cut into
	:type seq_length: int
	:return: Transformed data, Original data, MinMaxScaler object fitted on the original data
	:rtype: tuple(numpy.ndarray, numpy.ndarray, sklearn.preprocessing.MinMaxScaler)
	"""

	# todo: check if the feature engineering makes sense
	# # Load Google Data
	# x = np.loadtxt('https://github.com/firmai/tsgan/raw/master/alg/timegan/data/GOOGLE_BIG.csv', delimiter = ",",skiprows = 1)
	# #x = labels("https://github.com/firmai/tsgan/raw/master/alg/timegan/data/GOOGLE_BIG.csv")
	x = data
	# Build dataset
	dataX = []
	# Cut data by sequence length
	for i in range(0, len(x) - seq_length):
		_x = x[i:i + seq_length]
		dataX.append(_x)
	# Mix Data (to make it similar to i.i.d)
	idx = np.random.permutation(len(dataX))
	outputX = []
	for i in range(len(dataX)):
		outputX.append(dataX[idx[i]])
	X_train = np.stack(dataX)
	X_train = end_cond(X_train) #Adding the conditions
	x_x, scaler = MinMax(X_train[:,:,:-1])
	#x_x = np.c_[x_x, X_train[:,:,-1]]
	x_x = np.dstack((x_x, X_train[:,:,-1]))
	return x_x, X_train, scaler

def create_latent(dataX, batch_n = 1000, equal_batch=True, class_dim = 6, class_label = 4, z_dim = 50,z_val = 4, classed=True, noise="Normal" ):
  if classed==True:
    new = dataX[dataX[:,:,6] == class_label]
    new = np.reshape(new,(int(len(new)/24), 24, 7))[:,:,:-1]
    batch_num = new.shape[0]
    if not equal_batch:
      batch_num = batch_n
    print(batch_n)
    noise_class = np.zeros((batch_num, class_dim))
    noise_class[:,class_label] = 1
  else:
    noise_class = np.eye(class_dim)[np.random.choice(class_dim, batch_n)]
    new = dataX
    batch_num = batch_n
  if not equal_batch:
    batch_num = batch_n
  if noise=="Normal":
    z0 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
    z1 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
  elif noise=="z0_4":
    z0 = np.full((batch_num, z_dim), z_val)
    z1 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
  elif noise=="z1_4":
    z0 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
    z1 = np.full((batch_num, z_dim), z_val)
  return new, noise_class, z0, z1

d_recipe = {}
d_recipe["class0"] = [0, True, "Normal"]
d_recipe["class5"] = [5, True, "Normal"]
d_recipe["noise"] = [None, False, "Normal"]
d_recipe["z0_4"] = [None, False, "z0_4"]
d_recipe["z1_4"] = [None, False, "z1_4"]
d_recipe["org"] = d_recipe["noise"]
