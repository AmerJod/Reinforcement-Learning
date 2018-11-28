import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):

    def __init__(self, state_space, action_space, p_hidden_size, v_hidden_size):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space

        # Extract the state space and action space dimensions.
        input_dim = state_space.high.shape[0]
        output_dim = action_space.n

        self.critic = nn.Sequential(
            nn.Linear(input_dim, v_hidden_size),
            nn.ReLU(),
            nn.Linear(v_hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, p_hidden_size),
            nn.ReLU(),
            nn.Linear(p_hidden_size, output_dim),
            nn.Softmax(),
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_values = self.critic(state)

        return action_probs, state_values


class Agent(object):

    def __init__(self, network):
        self.network = network

        self.policy_reward = []
        self.policy_history = None
        self.value_history = None

    def select_move(self, state):
        pi_s, v_s = self.network(state)

        # Sample an action from this distribution.
        c_action = Categorical(pi_s)
        action = c_action.sample()

        log_action = c_action.log_prob(action).view(-1, 1)
        v_s = v_s.view(-1, 1)

        # Caches the log probabilities of each action for later use.
        if self.policy_history is None:
            self.policy_history = log_action

        else:
            self.policy_history = torch.cat([self.policy_history, log_action])

        if self.value_history is None:
            self.value_history = v_s

        else:
            self.value_history = torch.cat([self.value_history, v_s])

        return action

    def reset_history(self):
        self.policy_reward = []
        self.policy_history = None
        self.value_history = None


if __name__ == '__main__':

    # Set up the CartPole Environment.
    env = gym.make("CartPole-v0")

    # Retrieve the state space and action space objects for CartPole.
    state_space = env.observation_space
    action_space = env.action_space
    p_hidden_size = 16
    v_hidden_size = 16

    model = ActorCritic(state_space, action_space, p_hidden_size, v_hidden_size)
    agent = Agent(model)

    # Set up the loss function and optimiser for the NNs.
    optimiser = torch.optim.RMSprop(model.parameters())

    episodes = 100
    discount = 0.99
    for episode in range(episodes):
        agent.reset_history()
        state = torch.Tensor(env.reset())

        total_reward = 0
        done = False
        while not done:
            env.render()
            action = agent.select_move(state)
            next_state, reward, done, info = env.step(action.data.numpy())

            agent.policy_reward.append(reward)

            state = torch.Tensor(next_state)
            total_reward += reward

        # Cache the discounted rewards at each step of the episode
        rewards = []
        R = 0

        # Calculate the discounted reward at each step of the episode.
        for r in agent.policy_reward[::-1]:
            R = r + discount * R
            rewards.insert(0, R)

        # Normalise the rewards for stability.
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        # Retrieve the log probabilities of the actions over time.
        log_pi_t = agent.policy_history
        v_s_t = agent.value_history
        advantage = rewards - v_s_t

        policy_loss = (-log_pi_t * advantage.detach()).mean()

        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        print(policy_loss, 0.5 * value_loss, total_reward)

        optimiser.zero_grad()

        # Retrieve gradients of the actor network
        loss.backward()

        # Take a gradient descent step for the actor.
        optimiser.step()
