from train import TrainNet
from parser import parse_arg, parse_shape

class Predict(TrainNet):
	def valid(self):
		for i in range(self.data.len):
			y, x = self.data.get_for_test(i)
			x = x.reshape(x.shape[0], 1)
			pred = self.forward(x)
			if pred > 0.5:
				p = "B"
			else:p = "M"
			print("ID: ", self.data.ID[i], " Predict: ", pred[0,0], " Class: ", p)




if __name__ == "__main__":

	args = parse_arg()
	if not args.weight:
		print("Give me weights")
		exit(0)
	else:
		shape = parse_shape(args.weight)
		print("Shape: ", shape)
		net = Predict(shape, args.train)
		net.read(args.weight)
		net.valid()