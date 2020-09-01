import torch
import numpy as np
import torch.optim as opt
from models.explainer import MolDQN
from config.explainer import Args


class Agent(object):
    def __init__(self, num_input, num_output, device):

        Hyperparams = Args()
        REPLAY_BUFFER_CAPACITY = Hyperparams.replay_buffer_size

        self.device = device

        self.dqn, self.target_dqn = (
            MolDQN(num_input, num_output).to(self.device),
            MolDQN(num_input, num_output).to(self.device)
        )

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        import warnings
        warnings.filterwarnings("ignore")
        from baselines.deepq import replay_buffer
        warnings.filterwarnings("default")

        self.replay_buffer = replay_buffer.ReplayBuffer(REPLAY_BUFFER_CAPACITY)

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

        states, _, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        q_t = torch.zeros(batch_size, 1, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)

        for i in range(batch_size):
            state = (
                torch.FloatTensor(states[i])
                .reshape(-1, Hyperparams.fingerprint_length + 1)
                .to(self.device)
            )

            q_t[i] = self.dqn(state)

            next_state = (
                torch.FloatTensor(next_states[i])
                .reshape(-1, Hyperparams.fingerprint_length + 1)
                .to(self.device)
            )

            v_tp1[i] = torch.max(self.target_dqn(next_state))

        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)

        # get q values
        q_tp1_masked = (1 - dones) * v_tp1
        q_t_target = rewards + gamma * q_tp1_masked
        td_error = q_t - q_t_target

        q_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, pt in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                pt.data.mul_(polyak)
                pt.data.add_((1 - polyak) * p.data)

        return q_loss
