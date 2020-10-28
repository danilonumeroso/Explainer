import torch
import numpy as np
import torch.optim as opt
from models.explainer import MolDQN
from models.explainer.ReplayMemory import ReplayMemory
from config.explainer import Args


class Agent(object):
    def __init__(self, num_input, num_output, device):

        Hyperparams = Args()
        REPLAY_BUFFER_CAPACITY = Hyperparams.replay_buffer_size

        self.device = device
        self.num_input = num_input

        self.dqn, self.target_dqn = (
            MolDQN(num_input*2, num_output).to(self.device),
            MolDQN(num_input*2, num_output).to(self.device)
        )

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayMemory(REPLAY_BUFFER_CAPACITY)

        self.optimizer = getattr(opt, Hyperparams.optimizer)(
            self.dqn.parameters(), lr=Hyperparams.lr
        )

    def get_action(self, observations, epsilon_threshold):

        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).detach().numpy()

        return action

    def update_params(self, batch_size, gamma, polyak):

        Hyperparams = Args()

        experience = self.replay_buffer.sample(batch_size)

        # states = torch.tensor([S for S, *_ in experience]).float().reshape(-1, self.num_input).to(self.device)
        states = torch.stack([S for S, *_ in experience]).to(self.device)

        next_states = torch.stack([S for *_, S, _ in experience]).to(self.device)

        q, q_target = self.dqn(states), torch.max(self.target_dqn(next_states), dim=1).values.detach()

        q, q_target = q.to(self.device), q_target.to(self.device)


        rewards = torch.tensor([R for _, R, *_ in experience]).resize_as_(q).to(self.device)

        dones = torch.tensor([D for *_, D in experience]).resize_as_(q).to(self.device)

        q_target = rewards + gamma * (1 - dones) * q_target
        td_target = q - q_target

        loss = torch.where(
            torch.abs(td_target) < 1.0,
            0.5 * td_target * td_target,
            1.0 * (torch.abs(td_target) - 0.5),
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)

        return loss
