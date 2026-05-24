import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        """
        
        These lines define a learnable standard deviation for the Gaussian policy.
        Since Hopper has 3 continuous actions, sigma has 3 values, 
        one for each actuator.
        The raw sigma values are initialized at 0.5 and 
        later passed through softplus so the actual standard deviation is always positive.
        During training, the optimizer can update sigma,
        allowing the agent to control how much it explores around the mean action.
        
        """
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus #The standard deviation of a Gaussian distribution must always be positive.
        init_sigma = 0.5 # This sets the initial raw value of sigma.
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
        #torch.nn.Parameter tells PyTorch:This tensor is a trainable parameter of the model.

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)#This creates the Gaussian policy distribution.: π(a | s)= Normal(μ(s), σ)


        """
            Critic
        """
        # TASK 3: forward in the critic network

        
        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu', baseline_type="none", baseline_value=0.0,normalize_advantages=False,):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        # REINFORCE baseline configuration
        self.baseline_type = baseline_type
        self.baseline_value = baseline_value
        self.normalize_advantages = normalize_advantages

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        #This combines all stored action log probabilities into one tensor.
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        
        #This combines all stored action log probabilit0ies into one tensor.
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        
        #This creates a tensor of next states.
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        
        #This creates a tensor of all rewards.These rewards are used to compute discounted returns.
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        #This converts the done flags into a tensor.The done flag tells us whether the episode ended.
        done = torch.Tensor(self.done).to(self.train_device)

        #This clears the memory after converting everything into tensors.Because after updating the policy, the next episode should start fresh.
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # --------------------------------------------------
        # TASK 2: REINFORCE without baseline
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        # --------------------------------------------------

        # 1. Compute discounted returns G_t for each timestep
        returns = discount_rewards(rewards, self.gamma)

        # 2. Apply baseline if requested
        # Without baseline:
        #   advantage = G_t
        # With constant baseline:
        #   advantage = G_t - b
        if self.baseline_type == "none":
            advantages = returns

        elif self.baseline_type == "constant":
            advantages = returns - self.baseline_value

        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")
        
        # 3. Optional advantage normalization
        # Note:
        # Full mean-normalization can cancel the effect of a constant baseline.
        # Therefore, for the official constant-baseline comparison,
        # keep normalize_advantages=False.
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # 4. Policy gradient loss
        # REINFORCE objective: maximize log_prob(action) * return
        # PyTorch minimizes losses, so we use the negative sign.
        policy_loss = -(action_log_probs * advantages).sum()

        # 5. Backpropagation and optimizer step
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        return        


    # This function converts one environment state into an action.
    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        # This passes the state through the policy network.
        # The output is a Gaussian distribution: π(a | s) = Normal(μ(s), σ)
        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample() #During training, the agent samples an action with randomness to create exploration
            
            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            # action → motor command sent to Hopper
            # action_log_prob → log probability of that action under the current policy
            # If an action led to high return, increase its probability.
            # If an action led to low return, decrease its probability.

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

