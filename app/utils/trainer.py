from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers



class Estimator:
    @staticmethod
    def fit(X_t, y):

        embed_size = 128
        max_features = 6000
        model = Sequential()
        model.add(Embedding(max_features, embed_size))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(20, activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 100
        epochs = 3
        return model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    @staticmethod
    def predict(trained, X_te):
        prediction = trained.predict(X_te)
        y_pred = (prediction > 0.5)
        return y_pred