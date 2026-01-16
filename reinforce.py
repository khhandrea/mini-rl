import gymnasium as gym
import torch
from torch import cuda
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


class Args:
	training_step = 10000
	lr = 2e-4
	gamma = 0.98


class Policy(nn.Module):
	def __init__(self):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(4, 128),
			nn.ReLU(),
			nn.Linear(128, 2),
			nn.Softmax(dim=-1)
		)

	def forward(self, x):
		x = self.mlp(x)
		return x

	def get_action(self, state):
		action_prob = self.forward(state)
		distribution = Categorical(action_prob)
		action = distribution.sample()
		log_prob = distribution.log_prob(action)
		return action.item(), log_prob
		

def main():
	args = Args()
	device = "cuda" if cuda.is_available() else "cpu"
	writer = SummaryWriter("runs/REINFORCE")

	policy = Policy()
	policy.to(device)
	optimizer = optim.Adam(policy.parameters(), lr=args.lr)

	env = gym.make("CartPole-v1")

	for episode in range(args.training_step):
		state, _ = env.reset()
		replay_buffer = []
		done = False
		episode_return = 0.0

		while not done:
			state = torch.from_numpy(state).to(device)
			action, log_prob = policy.get_action(state)
			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated

			replay_buffer.append((log_prob, reward))
			state = next_state
			episode_return += reward

		# Calculate returns
		G = 0
		loss = 0
		for log_prob, reward in reversed(replay_buffer):
			G = reward + args.gamma * G
			loss += -log_prob * G

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		writer.add_scalar("Loss/train", loss.item(), episode + 1)
		writer.add_scalar("Score/return", episode_return, episode + 1)
		print(f"Return of episode {episode + 1}: {episode_return}")
	writer.close()


if __name__ == "__main__":
	main()
