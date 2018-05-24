from FNN.MyModel import MyModel
import tensorflow as tf
import numpy as np


class PredictMyModel(MyModel):
    @staticmethod
    def predict_on_model(loaded_model, image):
        image = np.expand_dims(image, axis=0)
        return loaded_model.predict(image)
