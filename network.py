import numpy as np

## TODO add new LOSS


class FullConnectNet():
	def __init__(self, size):
		self.num_of_layers = len(size)
		self.size = size
		self.biases = [np.random.normal(0, 0., [x, 1]) for x in size[1:]]
		self.weigts = [np.random.normal(0, 0., [x, y]) for x, y in zip(size[1:], size[:-1])]

	def forward(self, input):
		activation = input
		for b, w in zip(self.biases, self.weigts):
			s = w @ activation + b # sum
			activation = self._sigmoid(s)
		return activation

	def _sigmoid(self, arr):
		return 1 / (1 + np.exp(-arr))

	def _sigmoid_derivation(self, z):
		return self._sigmoid(z) * (1 - self._sigmoid(z))

	def loss(self, true, pred):
		return -(true * np.log(pred) + (1 - true) * np.log(1 - pred))

	def _loss_derivation(self, pred, true):
		return -(true - pred) / (pred * (1 - pred) + 0.0001)

	def backprop(self, input, grtruth):
		dbias = [np.zeros(b.shape) for b in self.biases]
		dweight = [np.zeros(w.shape) for w in self.weigts]

		activation = input
		activations = [input]
		sums = []

		# forward pass to create
		# 	nodes activation values

		for b, w in zip(self.biases, self.weigts):
			s = w @ activation + b  # sum
			activation = self._sigmoid(s)
			sums.append(s)
			activations.append(activation)


			# Gradients for LAST layer

		dbiastmp =	self._loss_derivation(activation[-1], grtruth) * \
			self._sigmoid_derivation(sums[-1])
		dbias[-1] = dbiastmp
		dweight[-1] = np.tensordot(dbiastmp, activations[-2].T, axes=0).\
			reshape((dbiastmp.shape[0],activations[-2].shape[0]))

			# Gradients for HIDDEN layer

		for l in range(2, self.num_of_layers):
			s = sums[-l]
			derivation = self._sigmoid_derivation(s)
			dbiastmp = (self.weigts[-l + 1].T @ dbiastmp) * derivation
			dbias[-l] = dbiastmp
			dweight[-l] = np.tensordot(dbiastmp, activations[-l-1].T, axes=0).\
				reshape((dbiastmp.shape[0],activations[-l-1].shape[0]))

		return (np.asarray(dbias), np.asarray(dweight), self.loss(grtruth, \
																  activations[-1]))
		## public function for tests ####


if __name__ == "__main__":

	net = FullConnectNet([2,3,3,2])
	net.backprop(np.array([1,1]), np.array([1,1]))


