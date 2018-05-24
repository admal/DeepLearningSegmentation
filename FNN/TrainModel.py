from FNN.MyModel import MyModel
from config import EPOCHS_COUNT


class TrainMyModel(MyModel):
    def train(self, x, y, x_val, y_val, model=None):
        if model is None:
            model = self.get_model()

        model.fit(
            x, y, batch_size=1, epochs=EPOCHS_COUNT, verbose=1, validation_data=(x_val, y_val)
        )
        model_json = model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")

        return model
    # model.save(self.model_dir)
