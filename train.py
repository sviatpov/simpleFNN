import numpy as np
from network import FullConnectNet
from dataloader import dataloader
from vizualizer import vizualizer
from parser import parse_arg, parse_shape

class TrainNet(FullConnectNet):

	def __init__(self, size, path, ln=1, nesterov=0):
		super(TrainNet, self).__init__(size)
		self.data = dataloader(path)
		self.ln = ln
		self.tmplosses = 10000000
		self.best_b = self.biases.copy()
		self.best_w = self.weigts.copy()
		self.vizualizer = vizualizer()
		self.gamma_nesterov = nesterov
		self.biases_nesterov = [np.random.normal(0, 0, [x, 1]) for x in size[1:]]
		self.weigts_nesterov = [np.random.normal(0, 0, [x, y]) for x, y in zip(size[1:], size[:-1])]

	def _udate_weight(self, start, size):
		db = np.asarray([np.zeros(b.shape) for b in self.biases])
		dw = np.asarray([np.zeros(w.shape) for w in self.weigts])
		losses = []
		for i in range(start, start+size):
			y, x = self.data.get_for_train(i)
			x = x.reshape(x.shape[0], 1)
			db_unit, dw_unit, loss = self.backprop(x, y)
			db += db_unit
			dw += dw_unit
			losses.append(loss)

		for w, deltaw, nest in zip(self.weigts, dw, self.weigts_nesterov):
			w -= (deltaw * self.ln / size + nest * self.gamma_nesterov)
		for b, deltab, nest in zip(self.biases, db, self.biases_nesterov):
			b -= (deltab * self.ln / size + nest * self.gamma_nesterov)
		self.weigts_nesterov = dw.copy()
		self.biases_nesterov = db.copy()

		return sum(losses) / len(losses)


	def train(self, batchsize=1, epoch=10):
		train_losses = []
		for e in range(epoch):
			for batch in range(0, len(self.data), batchsize):
				train_loss = self._udate_weight(batch, batchsize)
				train_losses.append(train_loss)
			valid_loss = self._valid()
			train_loss = sum(train_losses)/ len(train_losses)
			self.vizualizer.add((train_loss[0, 0], valid_loss[0, 0]))
			print("Epoch {}/{} -train loss: {}   -valid loss: {}".\
				  format(e, epoch, train_loss, valid_loss))


	def _valid(self):
		losses = []
		for i in self.data.ids_valid_split:
			y, x = self.data.get_for_test(i)
			x = x.reshape(x.shape[0], 1)
			pred = self.forward(x)
			loss = self.loss(y, pred)
			losses.append(loss)

		loss = sum(losses) / len(losses)
		if self.tmplosses > loss:
			self.tmplosses = loss
			self.best_b = self.biases.copy()
			self.best_w = self.weigts.copy()
		return loss

	def save(self):
		np.save('weights', np.asarray([np.asarray(self.biases), np.asarray(self.weigts), np.asarray(self.size)]))

	def save_best(self):
		np.save('weights', np.asarray([np.asarray(self.best_b), np.asarray(self.best_w), np.asarray(self.size)]))

	def read(self, path):
		L = np.load(path)
		i = 0
		for a in L[1]:
			assert a.shape == self.weigts[i].shape
			self.weigts[i] = np.asarray(a, dtype='float32')
			i += 1
		i = 0
		for a in L[0]:
			assert a.shape == self.biases[i].shape
			self.biases[i] = np.asarray(a, dtype='float32')
			i += 1

	def save_losses(self):
		self.vizualizer.save_png()


if __name__ == "__main__":

	args = parse_arg()
	shape = args.net_shape.strip().split(" ")
	arr = []
	arr.append(30)
	for i in shape:
		arr.append(int(i))
	arr.append(1)
	if args.nesterov:
		print("Optimizer: SGD")
	else:
		print("Optimizer: Nesterov")
	if not args.weight:
		print("Shape: ", arr)
		net = TrainNet(arr, args.train, args.lr, args.nesterov)
		net.train(args.batch, args.epoch)

	else:
		shape = parse_shape(args.weight)
		print("Shape: ", shape)
		net = TrainNet(shape, args.train, args.lr, args.nesterov)
		net.read(args.weight)
		net.train(args.batch, args.epoch)

	if args.best:
		net.save_best()
	else:
		net.save()
	net.save_losses()