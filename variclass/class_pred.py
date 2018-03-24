#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# Copyright (C) 2017 Angelo Falchetti
# All rights reserved.
# 
# This Work has been provided with no license
# whatsoever. You do not have the right to do
# anything with it. This includes (but is not
# limited to) copying, distributing, reproducing,
# executing, storing, preparing Derivative Works of,
# publicly displaying, publicly performing, sublicensing
# or using it in any other form.
# 
# Neither this entire notice nor any subset of it shall be
# construed to be a license of any kind under any circumstance.

from keras.layers import Input, Dense, LSTM, Dropout, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import plot_model, to_categorical
import numpy as np

seqlen    = 100
batchsize = 32
lambda_   = 0.5
memory    = 32

LastOutput = Lambda(lambda x: x[:, -1:, :], output_shape=lambda s: (s[0], 1, s[2]))

inputdata = Input(shape=(seqlen, 3))
lstm      = LSTM(memory, return_sequences=True)(inputdata)
# here we can play with multiple lstm layers
# ...lstm3 = LSTM(32, return_sequences=True)(lstm2)...
# memory size, LSTM(48, ...), LSTM(96, ...), LSTM(128, ...)
# and activations, LSTM(32, return_sequences=True, activation="relu")
dropped   = Dropout(0.5)(lstm)
predict   = TimeDistributed(Dense(1), input_shape=(seqlen, memory))(dropped)
last      = LastOutput(dropped)
classif   = Dense(1, activation="sigmoid")(last)

model = Model(inputs=[inputdata], outputs=[classif, predict])
plot_model(model, "model.png")

model.compile(optimizer="adam",
              loss=["binary_crossentropy", "mean_squared_error"],
              loss_weights=[1.0, lambda_])

# fake data
nsamples  = 1000
xreal     = np.random.random((nsamples, seqlen, 2))  # 2 -> (t, q(t))
classreal = np.random.randint(1, size=(nsamples, 1, 1)).astype("float32")
nextreal  = xreal[:, :, 1:2]

# input formatting
previous  = np.concatenate((xreal[:, 0:1, :], xreal[:, :-1, :]), axis=1)
dt        = xreal[:, :, 0:1] - previous[:, :, 0:1]
inputreal = np.concatenate((xreal[:, :, 0:1], previous[:, :, 1:2], dt), axis=2)

import ipdb;ipdb.set_trace()

model.fit(x=[inputreal], y=[classreal, nextreal], batch_size=batchsize, epochs=10)