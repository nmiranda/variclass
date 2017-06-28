# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pyfits
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.losses import mean_squared_error, binary_crossentropy
import theano

def loss_function(y_true, y_pred, _lambda=0.0000000000001):

	loss_class = binary_crossentropy(y_true[:,-1,:-1], y_pred[:,-1,:-1])
	#loss_class = mean_squared_error(y_true, y_pred)

	loss_pred = mean_squared_error(y_true[:,:,-1], y_pred[:,:,-1])

	return loss_class + _lambda * loss_pred

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	args = parser.parse_args()

	type_list = list()
	jd_list = list()
	q_list = list()
	jd_delta_list = list()
	q_pred_list = list()

	max_jd = 0

	fits_files = glob.glob(os.path.join(args.dir, '*.fits'))
	for fits_file in fits_files:
		this_fits = pyfits.open(fits_file, memmap=False)
		try:
			this_data = this_fits[1].data
		except TypeError:
			continue
		this_header = this_fits[0].header

		this_jd = this_data['JD']
		this_q = this_data['Q']

		if this_jd.shape[0] > max_jd:
			max_jd = this_jd.shape[0]

		type_list.append(this_header['TYPE_SPEC'])
		jd_list.append(this_jd[1:])
		q_list.append(this_q[:-1])
		jd_delta_list.append(this_jd[1:] - this_jd[:-1])
		q_pred_list.append(this_q[1:])
		this_fits.close()
	max_jd = max_jd-1

	data_X = np.full((len(jd_list), max_jd, 3), 0.)
	for i in xrange(len(jd_list)):
		data_X[i,:jd_list[i].shape[0],0] = jd_list[i]
		data_X[i,:jd_delta_list[i].shape[0],1] = jd_delta_list[i]
		data_X[i,:q_list[i].shape[0],2] = q_list[i]

	#scaler = MinMaxScaler(feature_range=(0, 1))
	#data_X[:,:,0] = scaler.fit_transform(data_X[:,:,0])

	label_encoder = LabelEncoder().fit(type_list)
	type_list = label_encoder.transform(type_list)
	data_Y = np.full((len(jd_list), max_jd, len(label_encoder.classes_)+1), 0.)
	for i, value in enumerate(type_list):
		data_Y[i,max_jd-1,value] = 1
		data_Y[i,:len(jd_list[i]),len(label_encoder.classes_)] = jd_list[i]

	#import ipdb;ipdb.set_trace()

	"""
	model = Sequential()
	model.add(LSTM(4, input_dim=3, return_sequences=True))
	model.add(Dense(len(label_encoder.classes_)+1))
	model.compile(loss=loss_function, optimizer='adam')
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	model.fit(data_X, data_Y, epochs=10, verbose=2)
	"""

	LastOutput = Lambda(lambda x: x[:, -1:, :], output_shape=lambda y: (y[0], 1, y[2]))

	_input = Input(shape=(len(jd_list), 3))
	lstm = LSTM(32, return_sequences=True)(_input)
	dropout = Dropout(0.5)(lstm)
	time_distributed = TimeDistributed(Dense(1), input_shape=(len(jd_list), 32))(dropout)
	last_output = LastOutput(dropout)
	dense = Dense(1, activation="sigmoid")(last_output)

	model = Model(inputs=[_input], outputs=[dense, time_distributed])
	plot_model(model, "model.png")

	model.compile(optimizer='adam', loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1.0, _lambda])

	model.fit(x=[inputreal], y=[classreal, nextreal], batch_size=batchsize, epochs=10)

	trainPredict = model.predict(data_X)

	print trainPredict[0]




if __name__ == '__main__':
    main()