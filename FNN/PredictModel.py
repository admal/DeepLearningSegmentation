from FNN.Model import Model
import tensorflow as tf
import numpy as np


class PredictModel(Model):
    def predict(self, image):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        image = np.expand_dims(image, axis=0)
        return loaded_model.predict(image)
