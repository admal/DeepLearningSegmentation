from FNN.Model import Model
from config import EPOCHS_COUNT


class TrainModel(Model):
	def train(self, x, y, x_val, y_val):
		model = self.get_model()

		model.fit(
			x,y, batch_size=4, epochs=EPOCHS_COUNT,verbose=1,validation_data=(x_val, y_val)
		)
		model_json = model.to_json()

		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("model.h5")
		# model.save(self.model_dir)
