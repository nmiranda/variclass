from keras.models import Model
from keras import optimizers
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense

class Convolutional(Model):
    """docstring for Convolutional"""

    def __init__(self, max_jd, input_dim=2, conv_filters=64, window_size=9, dropout_rate=0.25, dense_dim=64, dense_activation='sigmoid', learning_rate=0.001):

        # Init variables
        self.max_jd = max_jd
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.dense_activation = dense_activation
        #self.model_optimizer = optimizers.Adam()
        self.learning_rate = learning_rate
        self.model_optimizer = optimizers.SGD(lr=self.learning_rate)
        self.inputs = None
        self.outputs = None
        #self.model_optimizer = optimizers.SGD(lr=self.learning_rate)

        # Defining model
        # _input = Input(shape=(max_jd-1, input_dim))
        # conv_1 = Conv1D(conv_filters, window_size)(_input)
        # max_pool_1 = MaxPooling1D(window_size)(conv_1)
        # conv_2 = Conv1D(conv_filters, window_size)(max_pool_1)
        # max_pool_2 = MaxPooling1D(window_size)(conv_2)
        # conv_3 = Conv1D(conv_filters, window_size)(max_pool_2)
        # max_pool_3 = MaxPooling1D(window_size)(conv_3)
        # dropout_1 = Dropout(dropout_rate)(max_pool_3)
        # flatten = Flatten()(dropout_1)
        # dense_1 = Dense(dense_dim)(flatten)
        # dropout_2 = Dropout(dropout_rate)(dense_1)
        # dense_2 = Dense(1, activation=dense_activation)(dropout_2)
        self.init_model()

        #model = Model(inputs=[_input], outputs=[dense_2])
        super(Convolutional, self).__init__(inputs=self.inputs, outputs=self.outputs)

        #model_optimizer = SGD(lr=0.05)

        self.compile(optimizer=self.model_optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    
    def init_model(self):
        _input = Input(shape=(self.max_jd-1, self.input_dim))
        conv_1 = Conv1D(self.conv_filters, self.window_size)(_input)
        max_pool_1 = MaxPooling1D(self.window_size)(conv_1)
        conv_2 = Conv1D(self.conv_filters, self.window_size)(max_pool_1)
        max_pool_2 = MaxPooling1D(self.window_size)(conv_2)
        conv_3 = Conv1D(self.conv_filters, self.window_size)(max_pool_2)
        max_pool_3 = MaxPooling1D(self.window_size)(conv_3)
        dropout_1 = Dropout(self.dropout_rate)(max_pool_3)
        flatten = Flatten()(dropout_1)
        dense_1 = Dense(self.dense_dim, activation=self.dense_activation)(flatten)
        dropout_2 = Dropout(self.dropout_rate)(dense_1)
        dense_2 = Dense(1, activation=self.dense_activation)(dropout_2)

        self.inputs = [_input]
        self.outputs = [dense_2]

    def config_str(self):

        return \
        "Conv filters: " + str(self.conv_filters) + "\n" + \
        "Dense activation: " + str(self.dense_activation) + "\n" + \
        "Dense dim: " + str(self.dense_dim) + "\n" + \
        "Window size: " + str(self.window_size) + "\n" + \
        "Dropout rate: " + str(self.dropout_rate) + "\n" + \
        "Model optimizer: " + str(self.model_optimizer.__class__()) + "\n" + \
        "Optimizer config: " + str(self.model_optimizer.get_config()) + "\n"

    def config_dict(self):

        this_conf = dict()
        this_conf['conv_filters'] = self.conv_filters
        this_conf['dense_activation'] = self.dense_activation
        this_conf['dense_dim'] = self.dense_dim
        this_conf['window_size'] = self.window_size
        this_conf['dropout_rate'] = self.dropout_rate
        this_conf['model_arch'] = self.get_config()
        this_conf['learning_rate'] = self.learning_rate
        this_conf['model_optimizer'] = str(self.model_optimizer.__class__)
        this_conf['optimizer_config'] = self.model_optimizer.get_config()
        return this_conf

# class Convolutional_v1(Convolutional):
#     """docstring for Convolutional_v1"""

#     # def __init__(self, arg):
#     #     super(Convolutional_v1, self).__init__()
#     #     self.arg = arg

#     def init_model(self):

#         self.conv_filters = 32
#         #self.model_optimizer = optimizers.SGD(lr=10)
#         self.model_optimizer = optimizers.SGD(lr=self.learning_rate)

#         _input = Input(shape=(self.max_jd-1, self.input_dim))
#         conv = Conv1D(self.conv_filters, self.window_size)(_input)
#         max_pool = MaxPooling1D(self.window_size)(conv)
#         flatten = Flatten()(max_pool)
#         dense = Dense(1, activation=self.dense_activation)(flatten)

#         self.inputs = [_input]
#         self.outputs = [dense]

class Convolutional_v1(Convolutional):
    """docstring for Convolutional_v1"""

    # def __init__(self, arg):
    #     super(Convolutional_v1, self).__init__()
    #     self.arg = arg

    def init_model(self):

        #self.conv_filters = 32
        #self.learning_rate = 0.0001
        #self.model_optimizer = optimizers.SGD(lr=10)
        #self.model_optimizer = optimizers.SGD(lr=self.learning_rate)

        _input = Input(shape=(self.max_jd-1, self.input_dim))
        conv_1 = Conv1D(self.conv_filters, self.window_size)(_input)
        max_pool_1 = MaxPooling1D(self.window_size)(conv_1)
        conv_2 = Conv1D(self.conv_filters, self.window_size)(max_pool_1)
        max_pool_2 = MaxPooling1D(self.window_size)(conv_2)
        flatten = Flatten()(max_pool_2)
        dense = Dense(1, activation=self.dense_activation)(flatten)

        self.inputs = [_input]
        self.outputs = [dense]

class Convolutional_v2(Convolutional):
    """docstring for Convolutional_v1"""

    # def __init__(self, arg):
    #     super(Convolutional_v1, self).__init__()
    #     self.arg = arg

    def init_model(self):

        #self.conv_filters = 32
        #self.model_optimizer = optimizers.SGD(lr=10)
        #self.model_optimizer = optimizers.SGD(lr=self.learning_rate)

        _input = Input(shape=(self.max_jd-1, self.input_dim))
        conv_1 = Conv1D(self.conv_filters, self.window_size)(_input)
        conv_2 = Conv1D(self.conv_filters, self.window_size)(conv_1)
        max_pool_1 = MaxPooling1D(self.window_size)(conv_2)
        conv_3 = Conv1D(self.conv_filters, self.window_size)(max_pool_1)
        conv_4 = Conv1D(self.conv_filters, self.window_size)(conv_3)
        max_pool_2 = MaxPooling1D(self.window_size)(conv_4)
        flatten = Flatten()(max_pool_2)
        dense = Dense(1, activation=self.dense_activation)(flatten)

        self.inputs = [_input]
        self.outputs = [dense]

class Convolutional_v3(Convolutional):
    """docstring for Convolutional_v1"""

    # def __init__(self, arg):
    #     super(Convolutional_v1, self).__init__()
    #     self.arg = arg

    def init_model(self):

        #self.conv_filters = 32
        #self.model_optimizer = optimizers.SGD(lr=10)
        #self.model_optimizer = optimizers.SGD(lr=self.learning_rate)

        _input = Input(shape=(self.max_jd-1, self.input_dim))
        conv_1 = Conv1D(self.conv_filters, self.window_size)(_input)
        max_pool_1 = MaxPooling1D(self.window_size)(conv_1)
        conv_2 = Conv1D(self.conv_filters, self.window_size)(max_pool_1)
        max_pool_2 = MaxPooling1D(self.window_size)(conv_2)
        conv_3 = Conv1D(self.conv_filters, self.window_size)(max_pool_2)
        max_pool_3 = MaxPooling1D(self.window_size)(conv_3)
        flatten = Flatten()(max_pool_3)
        dense = Dense(1, activation=self.dense_activation)(flatten)

        self.inputs = [_input]
        self.outputs = [dense]