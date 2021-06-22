# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from .model import DQN


class Agent():
	"""
		RainbowDQN으로 구성된 Agent
		Double DQN, Multi-step, Distributional RL 가 구현되어있는 모듈이다.
	"""
	def __init__(self, args, env):
		try:
			self.action_space = int(env.action_space.n)
		except:
			self.action_space = int(env) # action space 직접 지정하고 싶을 때는 그냥 숫자 넣으면 되도록
		self.atoms = args.atoms # distributional RL
		self.Vmin = args.V_min # distributional RL
		self.Vmax = args.V_max # distributional RL
		self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # bin을 나눠놓는것
		self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1) # 한칸당 차이
		self.batch_size = args.batch_size 
		self.n = args.multi_step
		self.discount = args.discount
		self.norm_clip = args.norm_clip

		self.net = DQN(args, self.action_space).to(device=args.device)
		self.target_net = DQN(args, self.action_space).to(device=args.device)
		self.net.train()
		self.update_target_net()
		self.target_net.train()    
		
		for param in self.target_net.parameters():
			param.requires_grad = False 

		self.optimiser = optim.Adam(self.net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

	def reset_noise(self):
		self.net.reset_noise()

	def act(self, state): # action 출력하는 함수
		with torch.no_grad():
			return (self.net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

	def learn(self, mem):
		idxs, states, actions, returns, next_states, done_masks , weights = mem.sample(self.batch_size) # replay buffer에서 샘플링
		log_p_s = self.net(states, log=True)  # log probability (모든 action head 다 출력)
		log_p_sa = log_p_s[range(self.batch_size), actions]  # log probability (actions 에 들어있는 head로만 선별)

		with torch.no_grad():
			p_s = self.net(next_states) # n-step 뒤의 확률
			d_s = self.support.expand_as(p_s) * p_s  # distribution으로 변경
			argmax_indices_ns = d_s.sum(2).argmax(1)  # argmax index 찾기
			self.target_net.reset_noise()  
			p_s = self.target_net(next_states)  # target Q (probability) head 전체
			p_sa = p_s[range(self.batch_size), argmax_indices_ns]  # target Q probability (policy 로 봐도 될 듯)

			# Distribution RL 과정
			V = returns.unsqueeze(1) + done_masks * (self.discount ** self.n) * self.support.unsqueeze(0) # Bellman equation
			V = V.clamp(min=self.Vmin, max=self.Vmax)  # Distribution 범위로 clamping

			# 기존의 distribution (히스토그램) 과 비교했을 때 bin의 범위가 바뀌기 때문에 L2 projection을 통해 같은 범위의 histogram으로 만들어줘야함
			b = (V - self.Vmin) / self.delta_z  
			l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64) # lower, upper bound
			# 범위 넘어가는 것들은 보정해준다.
			l[(u > 0) * (l == u)] -= 1 
			u[(l < (self.atoms - 1)) * (l == u)] += 1
			m = states.new_zeros(self.batch_size, self.atoms)
			offset = torch.linspace(0, ( (self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
			m.view(-1).index_add_(0, (l + offset).view(-1), (p_sa * (u.float() - b)).view(-1))  # 양쪽으로 분배
			m.view(-1).index_add_(0, (u + offset).view(-1), (p_sa * (b - l.float())).view(-1))  # 양쪽으로 분배

		loss = -torch.sum(m * log_p_sa, 1)  # m : target, log_p_sa : input distribution -> cross entropy 사용
		self.net.zero_grad()
		(weights * loss).mean().backward()  # importance weight를 곱해준다.
		clip_grad_norm_(self.net.parameters(), self.norm_clip)  # Gradient clipping을 해준다.(너무 gradient가 크면 bad policy가 나와버린다.)
		self.optimiser.step()

		mem.update_priorities(idxs, loss.detach().cpu().numpy())  # PER에서 TD-error update해서 sampling importance update

	def update_target_net(self): # target network update
		self.target_net.load_state_dict(self.net.state_dict())

	def save(self, path, name='model.pth'): # save model
		torch.save(self.net.state_dict(), os.path.join(path, name))

	def load(self, path, name='model.pth'): # load model for test
		self.net.load_state_dict(torch.load(os.path.join(path,name)))

	def train(self): 
		self.net.train()

	def eval(self):
		self.net.eval()
