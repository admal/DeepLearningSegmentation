from FNN.Model import Model
import tensorflow as tf
import numpy as np


class PredictModel(Model):
    def predict_on_model(self,loaded_model, image):
        image = np.expand_dims(image, axis=0)
        return loaded_model.predict(image)
