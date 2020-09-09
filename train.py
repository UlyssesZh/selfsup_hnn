from argparse import ArgumentParser
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import load_dataset
from the_hnn import SelfSupHNN

def get_args():
	parser = ArgumentParser(description=None)
	parser.add_argument('--dataset_dir', type=str)
	parser.add_argument('--learn_rate', default=1e-3, type=float)
	parser.add_argument('--hidden_dim', default=200, type=int)
	parser.add_argument('--nonlinearity', default='tanh', type=str)
	parser.add_argument('--save_dir', default='path')
	parser.add_argument('--name', default='selfsup_hnn')
	parser.add_argument('--n', type=int)
	parser.add_argument('--log_dir', default='runs', type=str)
	return parser.parse_args()

total_i = 0
def log(writer, loss):
	global total_i
	writer.add_scalar('Loss/train', loss, total_i)
	total_i += 1

def train(writer, model, dataset, learn_rate):
	optim = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=1e-4)
	loss = model.loss(dataset['x1'], dataset['x2'], dataset['t'])
	optim.zero_grad()
	loss.backward()
	optim.step()
	log(writer, loss)
	return loss.item()

if __name__ == '__main__':
	args = get_args()
	if not args.dataset_dir:
		print('Specify dataset dir with "--dataset_dir"')
		exit()
	if not args.n:
		print('Specify DOF with "--n"')
		exit()
		
	writer = SummaryWriter(log_dir=args.log_dir)
	model = SelfSupHNN(args.n, args.hidden_dim, args.nonlinearity)
	
	if torch.cuda.is_available():
		torch.device('cuda')
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		model.cuda()
	torch.autograd.set_detect_anomaly(True)
	
	step = 0
	dataset_list = [load_dataset(f'{args.dataset_dir}/{dataset}')
	                for dataset in os.listdir(args.dataset_dir)]
	for epoch in range(160):
		for dataset in dataset_list:
			dataset = dataset_list[0]#
			loss = train(writer, model, dataset, args.learn_rate)
			step += 1
			if step % 200 == 0:
				print(step, '- loss:', loss)
			
	writer.close()
	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	torch.save(model.state_dict(), f"{args.save_dir}/{args.name}.tar")
