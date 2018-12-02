import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



class dataloader():
	def __init__(self, path):
		df = pd.read_csv(path, header=None)
		self.GT = np.asarray(df.values[:,1] == 'B', dtype="float32")
		self.ID = np.asarray(df.values[:, 0], dtype="float32")
		self.df = np.asarray(df.values[:, 2:], dtype="float32") # skip ID and Truth from data_frame
		self.std = self.df.std(axis=0)
		self.mean = self.df.mean(axis=0)
		self.df = (self.df - self.mean) / self.std
		self.len = self.df.shape[0]
		self.ids_train_split, self.ids_valid_split = train_test_split([i for i in range(self.len)], \
																	  	test_size=0.2, random_state=42)


	def __len__(self):
		return len(self.ids_train_split)

	def get_for_train(self, i):
		if i >= len(self):
			i = i - len(self)
		return self.GT[self.ids_train_split[i]], \
				self.df[self.ids_train_split[i]].copy()

	def get_for_test(self, i):
		return self.GT[i], \
			   self.df[i].copy()



if __name__ == "__main__":
	d = dataloader('/home/sviatoslavpovod/Downloads/data.csv')
	print(len(d))