import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plot_ts(generators,
            noise_params,
            show=False,
            step=0,
            model_name="gan"):
	"""Generate fake time series arrays and plot them
    For visualization purposes, generate fake time series arrays
    then plot them in a square grid
    # Arguments
        generators (Models): gen0 and gen1 models for
            fake time series arrays generation
        noise_params (list): noise parameters
            (label, z0 and z1 codes)
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save time series arrays
        model_name (string): Model name
    """

	gen0, gen1 = generators
	noise_class, z0, z1 = noise_params
	feature0 = gen0.predict([noise_class, z0])
	tss = gen1.predict([feature0, z1])


def plot_corr(df):
	f = plt.figure(figsize=(5, 5))
	plt.matshow(df, fignum=f.number)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=5, rotation=45)
	plt.yticks(range(df.shape[1]), df.columns, fontsize=5)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=5)
	plt.title('Correlation Matrix', fontsize=16)


def inverse_trans(train_data, scaler: MinMaxScaler):
	num_instances, num_time_steps, num_features = train_data.shape
	train_data = np.reshape(train_data, (-1, num_features))
	train_data = scaler.inverse_transform(X=train_data)
	train_data = np.reshape(train_data, (num_instances, num_time_steps, num_features))
	return train_data


def create_latent(dataX, batch_n=1000, equal_batch=True, class_dim=6, class_label=4, z_dim=50, z_val=4, classed=True,
                  noise="Normal"):
	if classed == True:
		new = dataX[dataX[:, :, 6] == class_label]
		new = np.reshape(new, (int(len(new) / 24), 24, 7))[:, :, :-1]
		batch_num = new.shape[0]
		if not equal_batch:
			batch_num = batch_n
		print(batch_n)
		noise_class = np.zeros((batch_num, class_dim))
		noise_class[:, class_label] = 1
	else:
		noise_class = np.eye(class_dim)[np.random.choice(class_dim, batch_n)]
		new = dataX
		batch_num = batch_n
	if not equal_batch:
		batch_num = batch_n
	if noise == "Normal":
		z0 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
		z1 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
	elif noise == "z0_4":
		z0 = np.full((batch_num, z_dim), z_val)
		z1 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
	elif noise == "z1_4":
		z0 = np.random.normal(scale=0.5, size=[batch_num, z_dim])
		z1 = np.full((batch_num, z_dim), z_val)
	return new, noise_class, z0, z1


def bootstrap_from_historical(data: pd.DataFrame, length: int) -> pd.DataFrame:
	"""
	:param data: historical data
	:param len: length of the new data
	:return: new data
	"""
	bootstrap_index = np.random.choice(range(len(data)), size=length, replace=True)
	bootstrap_sample_df = data.iloc[bootstrap_index]
	return bootstrap_sample_df
