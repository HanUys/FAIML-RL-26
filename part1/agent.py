import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(rewards, gamma):
    """Compute discounted returns G_t for one full episode.

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """
    discounted_rewards = torch.zeros_like(rewards)
    running_return = 0.0

    for t in reversed(range(rewards.size(-1))):
        running_return = rewards[t] + gamma * running_return
        discounted_rewards[t] = running_return

    return discounted_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # --------------------------------------------------
        # Actor network
        # --------------------------------------------------
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learnable standard deviation for the Gaussian policy.
        # Hopper has 3 continuous actions, so sigma has 3 values.
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(
            torch.zeros(self.action_space) + init_sigma
        )

        # --------------------------------------------------
        # Critic network
        # --------------------------------------------------
        # The critic estimates V(s), a scalar value for each state.
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        """Return action distribution and critic value.

        Actor:
            state -> Gaussian action distribution

        Critic:
            state -> V(s)
        """

        # Actor forward pass
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        # Critic forward pass
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic)

        return normal_dist, state_value


class Agent(object):
    def __init__(
        self,
        policy,
        device="cpu",
        algorithm="reinforce",
        baseline_type="none",
        baseline_value=0.0,
        normalize_advantages=False,
        critic_loss_coef=0.5,
        entropy_coef=0.01,
        ac_target_type="td",
    ):
        self.train_device = device
        self.policy = policy.to(self.train_device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=1e-3,
        )

        self.gamma = 0.99

        # Algorithm configuration
        self.algorithm = algorithm

        # REINFORCE configuration
        self.baseline_type = baseline_type
        self.baseline_value = baseline_value
        self.normalize_advantages = normalize_advantages

        # Actor-Critic configuration
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.ac_target_type = ac_target_type

        # Episode memory
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        """Update policy after one collected episode."""

        action_log_probs = torch.stack(
            self.action_log_probs,
            dim=0,
        ).to(self.train_device).squeeze(-1)

        states = torch.stack(
            self.states,
            dim=0,
        ).to(self.train_device).squeeze(-1)

        next_states = torch.stack(
            self.next_states,
            dim=0,
        ).to(self.train_device).squeeze(-1)

        rewards = torch.stack(
            self.rewards,
            dim=0,
        ).to(self.train_device).squeeze(-1)

        done = torch.tensor(
            self.done,
            dtype=torch.float32,
            device=self.train_device,
        )

        # Clear memory after moving data to tensors.
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        # --------------------------------------------------
        # TASK 2: REINFORCE
        # --------------------------------------------------
        if self.algorithm == "reinforce":
            # 1. Compute discounted returns G_t.
            returns = discount_rewards(rewards, self.gamma)

            # 2. Apply optional constant baseline.
            if self.baseline_type == "none":
                advantages = returns

            elif self.baseline_type == "constant":
                advantages = returns - self.baseline_value

            else:
                raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

            # 3. Optional advantage normalization.
            # For the official constant-baseline comparison, keep this False,
            # because full mean-normalization can cancel a constant baseline.
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # 4. Policy-gradient loss.
            # PyTorch minimizes, so we use the negative sign.
            policy_loss = -(action_log_probs * advantages).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            return policy_loss.item()

        # --------------------------------------------------
        # TASK 3: Actor-Critic
        # --------------------------------------------------
        elif self.algorithm == "actor_critic":
            # 1. Estimate V(s_t).
            _, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)

            # 2. Compute critic targets.
            if self.ac_target_type == "td":
                # One-step bootstrapped TD target:
                # target_t = r_t + gamma * V(s_{t+1}) if not done
                # target_t = r_t if done
                _, next_state_values = self.policy(next_states)
                next_state_values = next_state_values.squeeze(-1)

                targets = rewards + self.gamma * next_state_values.detach() * (
                    1.0 - done
                )

            elif self.ac_target_type == "mc":
                # Monte Carlo target:
                # target_t = G_t
                # This is less dependent on early inaccurate critic estimates.
                targets = discount_rewards(rewards, self.gamma)

            else:
                raise ValueError(f"Unknown ac_target_type: {self.ac_target_type}")

            # 3. Advantage:
            # A_t = target_t - V(s_t)
            advantages = targets - state_values

            # 4. Optional advantage normalization for the actor update.
            actor_advantages = advantages.detach()
            if self.normalize_advantages:
                actor_advantages = (
                    actor_advantages - actor_advantages.mean()
                ) / (actor_advantages.std() + 1e-8)

            # 5. Actor loss.
            actor_loss = -(action_log_probs * actor_advantages).sum()

            # 6. Critic loss.
            critic_loss = F.mse_loss(state_values, targets.detach())

            # 7. Entropy bonus.
            # This encourages exploration by keeping the policy stochastic.
            normal_dist, _ = self.policy(states)
            entropy = normal_dist.entropy().sum(dim=-1).mean()

            # 8. Total Actor-Critic loss.
            loss = (
                actor_loss
                + self.critic_loss_coef * critic_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def get_action(self, state, evaluation=False):
        """Convert one environment state into one action.

        During training:
            sample action from Gaussian policy

        During evaluation:
            use mean action
        """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:
            return normal_dist.mean, None

        action = normal_dist.sample()
        action_log_prob = normal_dist.log_prob(action).sum()

        return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(done)