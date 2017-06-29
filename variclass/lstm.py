# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pyfits
import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Lambda, Input, Dropout
from keras.losses import mean_squared_error, binary_crossentropy
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	args = parser.parse_args()

	input_dim = 3 # <----  t, dt, q(t-dt)
	input_dtype = 'float32'
	lstm_memory = 32
	dropout_rate = 0.5
	lambda_loss=0.0000000000001
	batchsize = 32

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

		if this_header['TYPE_SPEC'] == 'QSO':
			type_list.append(1)
		else:
			type_list.append(0)
		jd_list.append(this_jd[1:])
		q_list.append(this_q[:-1])
		jd_delta_list.append(this_jd[1:] - this_jd[:-1])
		q_pred_list.append(this_q[1:])
		this_fits.close()

	# Input data dimensions for matrix
	max_jd = max_jd-1
	num_samples = len(jd_list)

	data_X = np.full((num_samples, max_jd, input_dim), 0., dtype=input_dtype)
	for i in xrange(num_samples):
		data_X[i,:jd_list[i].shape[0],0] = jd_list[i]
		data_X[i,:jd_delta_list[i].shape[0],1] = jd_delta_list[i]
		data_X[i,:q_list[i].shape[0],2] = q_list[i]

	class_real = np.asarray(type_list, dtype=input_dtype)[..., np.newaxis, np.newaxis]
	next_real = np.full((num_samples, max_jd, 1), 0., dtype=input_dtype)
	for i in xrange(num_samples):
		next_real[i,:q_pred_list[i].shape[0],0] = q_pred_list[i]

	LastOutput = Lambda(lambda x: x[:, -1:, :], output_shape=lambda shape: (shape[0], 1, shape[2]))

	_input = Input(shape=(max_jd, input_dim))
	lstm = LSTM(lstm_memory, return_sequences=True)(_input)
	dropout = Dropout(dropout_rate)(lstm)
	time_distributed = TimeDistributed(Dense(1), input_shape=(max_jd, lstm_memory))(dropout)
	last_output = LastOutput(dropout)
	dense = Dense(1, activation="sigmoid")(last_output)

	model = Model(inputs=[_input], outputs=[dense, time_distributed])
	plot_model(model, "model.png")

	model.compile(optimizer='adam', loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1.0, lambda_loss])

	model.fit(x=[data_X], y=[class_real, next_real], batch_size=batchsize, epochs=10)

	trainPredict = model.predict(data_X)

	print type(trainPredict)
	print len(trainPredict)
	print type(trainPredict[0])


if __name__ == '__main__':
    main()