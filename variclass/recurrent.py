from keras.models import Model
from keras.layers import Lambda, Input, LSTM, Dense, Flatten
from keras.layers.wrappers import TimeDistributed
from keras import optimizers

class Recurrent(Model):
    """docstring for Recurrent"""

    def __init__(self, max_jd, lstm_memory=64, input_dim=2, lstm_activation='tanh', dense_activation='sigmoid', lambda_loss=0.05, learning_rate=0.0001):
        
        # Init variables
        self.lstm_memory = lstm_memory
        self.input_dim = input_dim
        self.lstm_activation = lstm_activation
        self.max_jd = max_jd
        self.dense_activation = dense_activation
        self.lambda_loss = lambda_loss
        self.learning_rate = learning_rate
        self.model_optimizer = optimizers.Adam()

        self.inputs = None
        self.outputs = None
        self.loss = None
        self.loss_weights = None

        # Defining model
        self.init_model()

        # _input = Input(shape=(self.max_jd-1, self.input_dim))
        # lstm_1 = LSTM(self.lstm_memory, return_sequences=True, activation=self.lstm_activation)(_input)
        # lstm_2 = LSTM(self.lstm_memory, return_sequences=True, activation=self.lstm_activation)(lstm_1)
        # select_predict = SelectPredict(lstm_2)
        # time_distributed = TimeDistributed(Dense(1), input_shape=(self.max_jd, self.lstm_memory))(select_predict)
        # select_classify = SelectClassify(lstm_2)
        # flatten = Flatten()(select_classify)
        # dense = Dense(1, activation=self.dense_activation)(flatten)

        # Initializing model object
        super(Recurrent, self).__init__(inputs=self.inputs, outputs=self.outputs)

        if len(self.outputs) == 1:
            self.loss = ["binary_crossentropy"]
        elif len(self.outputs) == 2:
            self.loss = ["binary_crossentropy", "mean_squared_error"]
            loss_weights=[1.0, lambda_loss]

        self.compile(optimizer=self.model_optimizer, loss=self.loss, loss_weights=self.loss_weights, metrics=['accuracy'])

    def init_model(self):

        # Classification and prediction selection custom layers
        SelectPredict = Lambda(lambda x: x[:, :, :self.lstm_memory/2], output_shape=lambda shape: (shape[0], shape[1], self.lstm_memory/2))
        SelectClassify = Lambda(lambda x: x[:, :, self.lstm_memory/2:], output_shape=lambda shape: (shape[0], shape[1], self.lstm_memory/2))

        # Model structure
        _input = Input(shape=(self.max_jd-1, self.input_dim))
        lstm_1 = LSTM(self.lstm_memory, return_sequences=True, activation=self.lstm_activation)(_input)
        lstm_2 = LSTM(self.lstm_memory, return_sequences=True, activation=self.lstm_activation)(lstm_1)
        select_predict = SelectPredict(lstm_2)
        time_distributed = TimeDistributed(Dense(1), input_shape=(self.max_jd, self.lstm_memory))(select_predict)
        select_classify = SelectClassify(lstm_2)
        flatten = Flatten()(select_classify)
        dense = Dense(1, activation=self.dense_activation)(flatten)

        self.inputs = [_input]
        self.outputs = [dense, time_distributed]

    def config_str(self):

        return \
        "Lstm memory: " + str(self.lstm_memory) + "\n" + \
        "Lstm activation: " + str(self.lstm_activation) + "\n" + \
        "Dense activation: " + str(self.dense_activation) + "\n" + \
        "Lambda loss: " + str(self.lambda_loss) + "\n" + \
        "Optimizer: " + str(self.model_optimizer.__class__()) + "\n" + \
        "Optimizer config: " + str(self.model_optimizer.get_config()) + "\n"

    def config_dict(self):

        this_conf = dict()
        this_conf['lstm_memory'] = self.lstm_memory
        this_conf['lstm_activation'] = self.lstm_activation
        this_conf['dense_activation'] = self.dense_activation
        this_conf['lambda_loss'] = self.lambda_loss
        this_conf['learning_rate'] = self.learning_rate
        this_conf['model_optimizer'] = str(self.model_optimizer.__class__)
        this_conf['optimizer_config'] = self.model_optimizer.get_config()
        return this_conf

class Recurrent_v1(Recurrent):
    """docstring for Recurrent_v1"""

    def init_model(self):

        self.lstm_memory = 64

        # Classification and prediction selection custom layers
        SelectPredict = Lambda(lambda x: x[:, :, :self.lstm_memory/2], output_shape=lambda shape: (shape[0], shape[1], self.lstm_memory/2))
        SelectClassify = Lambda(lambda x: x[:, :, self.lstm_memory/2:], output_shape=lambda shape: (shape[0], shape[1], self.lstm_memory/2))

        # Model structure
        _input = Input(shape=(self.max_jd-1, self.input_dim))
        lstm = LSTM(self.lstm_memory, return_sequences=True, activation=self.lstm_activation)(_input)
        select_predict = SelectPredict(lstm)
        time_distributed = TimeDistributed(Dense(1), input_shape=(self.max_jd, self.lstm_memory))(select_predict)   
        select_classify = SelectClassify(lstm)
        flatten = Flatten()(select_classify)
        dense = Dense(1, activation=self.dense_activation)(flatten)

        self.inputs = [_input]
        self.outputs = [dense, time_distributed]

class Recurrent_v0(Recurrent):
    """docstring for Recurrent_v0"""

    def init_model(self):

        self.lstm_memory = 32
        self.model_optimizer = optimizers.SGD(lr=self.learning_rate)

        # Model structure
        _input = Input(shape=(self.max_jd-1, self.input_dim))
        lstm_1 = LSTM(self.lstm_memory, return_sequences=True, activation=self.lstm_activation)(_input)
        lstm_2 = LSTM(self.lstm_memory, return_sequences=True, activation=self.lstm_activation)(lstm_1)
        flatten = Flatten()(lstm_2)
        dense = Dense(1, activation=self.dense_activation)(flatten)

        self.inputs = [_input]
        self.outputs = [dense]