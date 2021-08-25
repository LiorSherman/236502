from tensorflow.keras.models import load_model


class Trainer:
    def __init__(self, model):
        self.model = model

    def fit(self, inputs, targets, batch_size, epochs=50):
        # train the model
        self.model.fit(inputs, targets, epochs=epochs, batch_size=batch_size)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(f"{path}/model.h5")
        self.model.mapping_file = f"{path}/mapping.json"
