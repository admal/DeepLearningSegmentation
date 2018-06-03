from SegNet.MyModel import MyModel
import numpy as np


class PredictMyModel(MyModel):
    @staticmethod
    def predict_on_model(loaded_model, image):
        image = np.expand_dims(image, axis=0)
        return loaded_model.predict(image)
