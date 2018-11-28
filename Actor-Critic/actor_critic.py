import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    '''
    A class representing the Actor-Critic model. Stores both the
    actor (policy) network and the critic (value) network.
    '''

    def __init__(self, state_space, action_space, p_hidden_size, v_hidden_size):
        '''
        Arguments:
            state_space: the StateSpace object of the environment
            action_space: the ActionSpace object of the environment
            p_hidden_size: the number of neurons in the hidden layer of the policy network.
            v_hidden_size: the number of neurons in the hidden layer of the value network.
        '''
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space

        # Extract the state space and action space dimensions.
        input_dim = state_space.high.shape[0]
        output_dim = action_space.n

        # Set up the critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, v_hidden_size),
            nn.ReLU(),
            nn.Linear(v_hidden_size, 1)
        )

        # Set up the actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, p_hidden_size),
            nn.ReLU(),
            nn.Linear(p_hidden_size, output_dim),
            nn.Softmax(),
        )

    def forward(self, state):
        '''
        Arguments:
            state: the current state of the environment

        Returns:
            action_probs: the action probabilities
            state_values: the value of the state
        '''
        action_probs = self.actor(state)
        state_values = self.critic(state)

        return action_probs, state_values


class Agent(object):
    '''
    A class that represents an action-critc agent. Deals with
    the selection of action, given a state, through the interaction
    with the actor and critic networks.
    '''

    def __init__(self, network):
        '''
        Arguments:
            network: the actor-critic networks
        '''
        self.network = network

        # Used to cache experience from the environment.
        self.policy_reward = []
        self.policy_history = None
        self.value_history = None

    def select_move(self, state):
        '''
        Selects an action, given a state.

        Arguments:
            state: the current state of the environment

        Returns:
            action: the selected action, given the state.
        '''

        # Retrieves the action-probabilities and value of a state,
        # given the state.
        pi_s, v_s = self.network(state)

        # Sample an action from this distribution.
        c_action = Categorical(pi_s)
        action = c_action.sample()

        # Retrieves the log probability of an action.
        log_action = c_action.log_prob(action).view(-1, 1)
        v_s = v_s.view(-1, 1)

        # Caches the log probabilities of each action for later use.
        if self.policy_history is None:
            self.policy_history = log_action

        else:
            self.policy_history = torch.cat([self.policy_history, log_action])

        # Caches the value of each state for later use.
        if self.value_history is None:
            self.value_history = v_s

        else:
            self.value_history = torch.cat([self.value_history, v_s])

        return action

    def reset_history(self):
        '''
        Resets the policy and value history of the agent.
        '''
        self.policy_reward = []
        self.policy_history = None
        self.value_history = None