import json
import os
import random

import torch
from torchdiffeq import odeint

from hamiltonian import dx_fun

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

def generate_dataset(save_path, hamiltonian, start_t, end_t, samples, x_min, x_max, seed=None, id=None):
	if seed:
		torch.manual_seed(seed)
	n = len(x_min) // 2
	fun = dx_fun(hamiltonian)
	t_span = [start_t, end_t]
	x1 = [(x_min + torch.rand(n*2) * (x_max - x_min)).tolist() for _ in range(samples)]
	dataset = {'t': t_span, 'x1': x1,
	           'x2': odeint(fun, torch.tensor(x1, requires_grad=True), torch.tensor(t_span))[-1].tolist()}
	with open(save_path, "w") as f:
		json.dump(dataset, f)
	print(f"{id}: finished")
