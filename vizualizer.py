
import matplotlib.pyplot as plt


class vizualizer():
	def __init__(self):
		self.loss_train = []
		self.loss_valid = []

	def save_png(self):
		if len(self.loss_train) == 0:
			return
		step = [i for i in range(len(self.loss_train))]
		plt.xlabel("EPOCH")
		plt.ylabel("Loss")
		plt.plot(step, self.loss_train)
		plt.savefig("train_losses.png")
		plt.close()
		plt.xlabel("EPOCH")
		plt.ylabel("Loss")
		plt.plot(step, self.loss_valid)
		plt.savefig("valid_losses.png")
		plt.close()

	def add(self, i):
		self.loss_train.append(i[0])
		self.loss_valid.append(i[1])