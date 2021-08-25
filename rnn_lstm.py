import tensorflow.keras as keras
import os
import datetime
from shutil import copyfile


class RnnLstm:
    def __init__(self, output_units, num_units, lr, mapping_file, loss="sparse_categorical_crossentropy",
                 activation="softmax"):
        """Builds and compiles model
            :param output_units (int): Num output units
            :param num_units (list of int): Num of units in hidden layers
            :param loss (str): Type of loss function to use
            :param lr (float): Learning rate to apply
            :return model (tf model): Where the magic happens :D
            """
        self.mapping_file = mapping_file
        # create the model architecture
        input = keras.layers.Input(shape=(None, output_units))
        x = keras.layers.LSTM(num_units[0])(input)
        x = keras.layers.Dropout(0.2)(x)

        output = keras.layers.Dense(output_units, activation=activation)(x)

        model = keras.Model(input, output)

        # compile model
        model.compile(loss=loss,
                      optimizer=keras.optimizers.Adam(learning_rate=lr),
                      metrics=["accuracy"])

        model.summary()

        self.model = model

    def fit(self, inputs, targets, batch_size, epochs=50):
        self.model.fit(inputs, targets, batch_size, epochs)

    def save(self, path):
        dir_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        full_path = path + "/" + dir_name
        os.makedirs(full_path)

        self.model.save(f"{full_path}/model.h5")
        copyfile(self.mapping_file, f"{full_path}/mapping.json")
        print(f"Model and mapping files saved to {full_path}")

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
