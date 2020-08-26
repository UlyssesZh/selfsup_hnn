import json
import os
import random

def load_dataset(path):
	with open(path) as fp:
		return json.load(fp)

def create_dataset(dir, save_path, samples_per_trajectory):
	def get_samples(filename):
		with open(f"{dir}/{filename}") as data_file:
			data = json.load(data_file)
			indices = random.sample(range(len(data)), samples_per_trajectory)
			indices.sort()
			return json.dumps([data[i] for i in indices])
	with open(save_path, 'w') as save_file:
		save_file.write('[')
		save_file.write(','.join([get_samples(filename) for filename in os.listdir(dir)]))
		save_file.write(']')
