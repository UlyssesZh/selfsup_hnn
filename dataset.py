import json
import os
import random

import torch
from torchdiffeq import odeint

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

def generate_dataset(save_path, hamiltonian, time_interval, samples_per_trajectory, x_min, x_max, trajectories, seed=None, id=None):
	if seed:
		torch.manual_seed(seed)
	n = len(x_min) // 2
	def fun(t, x):
		tx = torch.cat([torch.tensor([t]), x])
		tx.requires_grad_(True)
		hamiltonian(tx).backward()
		_, partial_q, partial_p = torch.split(tx.grad, [1, n, n])
		return torch.cat([partial_p, -partial_q])
	x0_span = x_max - x_min
	dataset = []
	for i in range(trajectories):
		x0 = x_min + x0_span * torch.rand(n)
		t_span = time_interval * torch.rand(samples_per_trajectory)
		t_span = t_span.sort().values
		x_span = odeint(fun, x0, t_span)
		dataset.append([{'t': t_span[j].item(), 'x': x_span[j].tolist()} for j in range(samples_per_trajectory)])
		if i % 10 == 9:
			print(f"{id}: {i+1} trajectories finished")
	with open(save_path, "w") as f:
		json.dump(dataset, f)
	print(f"{id}: finished")
