import numpy as np
import torch
from torch import nn

class PolicyNetwork(nn.Module):
	'''
	A class that represents the policy network, takes in a 
	state and produces the action-values of taking each
	valid action in that state.

	'''
	def __init__(self, state_space, action_space, hidden_size):
		'''
		Arguments:
			state_space: the StateSpace object of the environment
			action_space: the ActionSpace object of the environment
			hidden_size: the number of neurons in the hidden layer
		'''
		super().__init__()

		self.state_space = state_space
		self.action_space = action_space

		# Extract the state space and action space dimensions.
		input_dim = state_space.high.shape[0]
		output_dim = action_space.n

		# Define the architecture of the neural network.
		self.fc = nn.Sequential(
				nn.Linear(input_dim, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, output_dim)
			)

	def forward(self, state):
		'''
		Calculates the forward pass of the neural network.

		Arguments: 
			state: the state of the environments

		Returns: 
			output: the corresponding action-values from the network.
		'''
		output = self.fc(state)

		return output

class QAgent(object):
	'''
	A class representing an agent that follows the Q-learning 
	algorithm for learning an optimal policy.
	'''
	def __init__(self, policy_network, target_network, epsilon):
		'''
		Arguments:
			policy_network: the network that is updated at each step and 
							which the current action-value of a state is computed.
			target_network: the network that is used to construct the target action-values
							and is synchronised periodically.
			epsilon: determines the amount of exploration in e-greedy.
		'''
		self.policy_network = policy_network
		self.target_network = target_network
		self.epsilon = epsilon

		# Instantiate memory buffer.
		self.memory = []
		
	def select_action(self, state, action_space):
		'''
		Given a state, chooses an action in the action space
		according to e-greedy of the action-value function.

		Arguments:
			state: the state of the environment
			action_space: the action space of the environment.
		'''

		# Generate a random uniform between 0 and 1
		rand = np.random.rand()
	
		# If rand is less than eps, choose random action in action space.
		if rand < self.epsilon:
			action = np.random.randint(action_space.n)

		# Else, choose action greedily according to action-value function.
		else:
			output = self.policy_network.forward(state)
			action = torch.argmax(output)

		action = torch.tensor(action).view(-1, 1)

		return action