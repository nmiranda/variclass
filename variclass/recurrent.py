from keras.models import Model
from keras.layers import Lambda, Input, LSTM, Dense, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

class Recurrent(Model):
    """docstring for Recurrent"""

    def __init__(self, max_jd, lstm_memory=64, input_dim=2, lstm_activation='tanh', dense_activation='sigmoid', lambda_loss=0.05):
        
        # Init variables
        self.lstm_memory = lstm_memory
        self.input_dim = input_dim
        self.lstm_activation = lstm_activation
        self.max_jd = max_jd
        self.dense_activation = dense_activation
        self.lambda_loss = lambda_loss
        self.model_optimizer = Adam()

        # Defining model

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

        # Initializing model object
        super(Recurrent, self).__init__(inputs=[_input], outputs=[dense, time_distributed])

        self.compile(optimizer=self.model_optimizer, loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1.0, lambda_loss], metrics=['accuracy'])

    def config_str(self):

        return \
        "Lstm memory: " + str(self.lstm_memory) + "\n" + \
        "Lstm activation: " + str(self.lstm_activation) + "\n" + \
        "Dense activation: " + str(self.dense_activation) + "\n" + \
        "Lambda loss: " + str(self.lambda_loss) + "\n" + \
        "Optimizer: " + str(self.model_optimizer.__class__()) + "\n" + \
        "Optimizer config: " + str(self.model_optimizer.get_config()) + "\n"
