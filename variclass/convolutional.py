from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense

class Convolutional(Model):
    """docstring for Convolutional"""

    def __init__(self, max_jd, input_dim=2, conv_filters=32, window_size=5, dropout_rate=0.25, dense_dim=64, dense_activation='sigmoid'):

        # Init variables
        self.max_jd = max_jd
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.dense_activation = dense_activation
        self.model_optimizer = Adam()

        # Defining model
        _input = Input(shape=(max_jd-1, input_dim))
        conv_1 = Conv1D(conv_filters, window_size)(_input)
        max_pool_1 = MaxPooling1D(window_size)(conv_1)
        conv_2 = Conv1D(conv_filters, window_size)(max_pool_1)
        max_pool_2 = MaxPooling1D(window_size)(conv_2)
        conv_3 = Conv1D(conv_filters, window_size)(max_pool_2)
        max_pool_3 = MaxPooling1D(window_size)(conv_3)
        dropout_1 = Dropout(dropout_rate)(max_pool_3)
        flatten = Flatten()(dropout_1)
        dense_1 = Dense(dense_dim)(flatten)
        dropout_2 = Dropout(dropout_rate)(dense_1)
        dense_2 = Dense(1, activation=dense_activation)(dropout_2)

        #model = Model(inputs=[_input], outputs=[dense_2])
        super(Convolutional, self).__init__(inputs=[_input], outputs=[dense_2])

        #model_optimizer = SGD(lr=0.05)

        self.compile(optimizer=self.model_optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    
    def config_str(self):

        return \
        "Conv filters: " + str(self.conv_filters) + "\n" + \
        "Dense activation: " + str(self.dense_activation) + "\n" + \
        "Dense dim: " + str(self.dense_dim) + "\n" + \
        "Window size: " + str(self.window_size) + "\n" + \
        "Dropout rate: " + str(self.dropout_rate) + "\n" + \
        "Model optimizer: " + str(self.model_optimizer.__class__()) + "\n" + \
        "Optimizer config: " + str(self.model_optimizer.get_config()) + "\n"