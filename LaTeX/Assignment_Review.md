# Assignment 1
```python
# Gym Environment Semantics
env = gym.make(env_name)
obs, info = env.reset(seed=seed)
action = policy(obs)
observation, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
transitions = self.env.unwrapped.P[state][act]
for prob, next_state, reward, done in transitions:
  ...
```
```python
# Policy Iteration: Note that train() is updating value and then policy
random_mapping = np.random.randint(0, self.action_dim, size=(self.obs_dim))
self.policy: Callable = lambda obs: random_mapping[obs]
# Update Value Function
while True:
    old_table = self.value_table.copy()
    for state in range(self.obs_dim):
      action = self.policy(state)
      state_value = 0
      for transition in self._get_transitions(state, action):
        # Unpack transitions (omitted here)
        if done:
          state_value += prob * reward
        else:
          state_value += prob * (reward + self.gamma * old_table[next_state])
      self.value_table[state] = state_value
    if np.abs(old_table - self.value_table).sum() < self.eps:
      break
# Update Policy 
policy_table = np.zeroes([self.obs_dim, ], dtype=int)
for state in range(self.obs_dim):
  state_action_values = [0] * self.action_dim
  for action in range(self.action_dim):
    value = 0 
    for transition in self._get_transitions(state, action):
      # Omit unpacking logic here
      if done:
        value += prob * reward
      else:
        value += prob * (reward + self.gamma * self.value_table[next_state])
    state_action_values[action] = value
  policy_table[state] = np.argmax(state_value)
self.policy = lambda obs: policy_table[obs]
```
```python
# Value Iteration: Note that training is just updating value function
# Update Value Function
old_table = self.value_table.copy()
for state in range(self.obs_dim):
  state_value = 0
  state_action_values = [0] * self.action_dim
  for action in range(self.action_dim):
    value = 0
    for transition in self._get_transitions(state, action):
      # Omit unpacking logic here
      if done:
        value += prob * reward
      else:
          value += prob * (reward + self.gamma * old_table[next_state])
      state_action_values[action] = value
    state_action_valuesp[action] = value
  self.value_table[state] = max(state_action_values)
```
# Assignment 2
```python
# Q-Learning
# Epsilon Greedy Policy
if np.random.uniform(0, 1) < self.eps:
  action = self.env.action_space.sample()
else:
  action = np.argmax(self.table[obs])
return action
# Training
for t in range(self.max_episode_length):
  act = self.compute_action(obs) # Epsilon-Greedy
  next_obs, reward, terminated, truncated, info = self.env.step(act)
  done = terminated or truncated
  td_error = reward + self.gamma * np.max(self.table[next_obs]) - self.table[obs][act] 
  new_value = self.table[obs][act] + self.learning_rate * td_error
  self.table[obs][act] = new_value
  obs = next_obs
  if done:
    break
```
```python
# DQN
# ExperienceReplayMemory is a dequeue with push and sample methods
# Pytorch Model Syntax
self.action_value = nn.Sequential(
  nn.Linear(num_inputs, hidden_units),
  nn.ReLU(),
  nn.Linear(hidden_units, hidden_units),
  nn.ReLU(),
  nn.Linear(hidden_units, num_outputs)
)
# DQN Setup
# Set both network and target networks to eval mode (omitted here)
self.network = PytorchModel(self.obs_dim, self.act_dim, self.hidden_dim)
self.target_network = PytorchModel(self.obs_dim, self.act_dim)
self.target_network.load_state_dict(self.network.state_dict())
self.loss = nn.MSELoss()
# Training Function Snippets
act = self.compute_action(processed_obs) # Epsilon-Greedy
for t in range(self.max_episode_length):
  next_obs, reward, truncated, terminated, truncated, info = self.env.step(act)
  done = terminated or truncated
  self.memory.push((processed_obs, act, reward, next_processed_obs, done))
  act = self.compute_action(next_processed_obs)
  batch = self.memory.sample(self.batch_size)
  # Convert batches to tensors using np.stack and to_tensor - omitted
  with torch.no_grad():
    Q_t_plus_one = self.target_network(next_state_batch).max(dim=1)[0] 
    Q_objective = reward_batch.squeeze(0) + (1 - done_batch.squeeze(0)) * self.gamma * Q_t_plus_one
  
  self.network.train()
  Q_t = self.network(state_batch).gather(1, action_batch.long().squeeze(0).unsqueeze(1)).squeeze(1)
  self.optimizer.zero_grad()
  loss = self.loss(Q_t, Q_objective)
  loss.backward()
  self.optimizer.step()
  self.network.eval()

  if (update_weights): # omit log
    self.target_network.load_state_dict(self.network.state_dict())
    self.target_network.eval()
```
```python
# Policy Gradient Network Snippets
def forward(self, obs):
  logit = self.network(obs)
  dist = torch.distributions.Categorical(logits=logit)
  action = dist.sample()
  return action
def log_prob(self, obs, act):
  logit = self.network(obs)
  dist = torch.distributions.Categorical(logits=logit)
  log_prob = dist.log_prob(act)
  return log_prob
# Update
self.network.train()
self.optimizer.zero_grad()
log_probs = self.compute_log_probs(flat_obs, flat_act)
loss = -torch.mean(log_probs * advantages)
loss.backward()
torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config["clip_gradient"])
self.optimizer.step()
self.network.eval()
# Sample collection logic 
while iter_timesteps <= self.config["train_batch_size"]
  ...
  while True:
    act = self.compute_actions(obs)
    next_obs, reward, terminated, truncated, step_info = self.env.step(act)
    # Append everything
# Sample Processing
for reward_list in samples["reward"]
  ...
  for t in reversed(range(len(reward_list))):
    Q = reward_list[t] + self.config["gamma"] * Q
    returns[t] = Q
  values.append(returns) # technically values are advantages here
```
# Assignment 3
```python
# TD3 Actor (Continuous)
def forward(self, state):
  a = F.relu(self.l1(state))
  a = F.relu(self.l2(a))
  return self.max_action * torch.tanh(self.l3(a))
# TD3 Critic
def forward(self, state, action):
    sa = torch.cat([state, action], 1)
    q1 = F.relu(self.l1(sa))
    q1 = F.relu(self.l2(q1))
    q1 = self.l3(q1)
    q2 = F.relu(self.l4(sa))
    q2 = F.relu(self.l5(q2))
    q2 = self.l6(q2)
    return q1, q2
# TD3 Trainer Snippets (Actor, Actr Target, Critic, and Critic Targets)
state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
with torch.no_grad():
  noise = (torch.normal(0, self.policy_noise, size=action.shape).to(device)).clamp(-self.noise_clip, self.noise_clip)
  next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
  target_Q1, target_Q2 = self.critic_target(next_state, next_action)
  target_Q = torch.min(target_Q1, target_Q2)
  target_Q = reward + not_done * self.gamma * target_Q
current_Q1, current_Q2 = self.critic(state, action)
critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
self.critic_optimizer.zero_grad()
critic_loss.backward()
self.critic_optimizer.step()
if (update_actor):
  actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
  self.actor_optimizer.zero_grad()
  actor_loss.backward()
  self.actor_optimizer.step()
  # Update target models here (rolling update)
  # e.g. target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```
```python
# PPO
# Rollout Storage GAE Advantage Computing
self.value_preds[-1] = next_value
gae = 0
for step in reversed(range(self.rewards.size(0))):
  delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
  gae = delta + gamma * self.gae_lambda * self.masks[step + 1] * gae
  self.returns[step] = gae + self.value_preds[step]
# PPO Trainer Snippets
def compute_action(self, obs, deterministic=False):
  actions, action_log_probs = None, None
  if self.discrete:
    logits, values = self.model(obs)
    dist = Categorical(logits = logits)
    actions = dist.probsl.argmax(dim=-1) if determinstic else dist.sample()
    action_log_probs = dist.log_prob(actions)
  else:
    means, log_std, values = self.model(obs)
    std = log_std.exp()
    dist = torch.distributions.Normal(means, std)
    actions = means if deterministic else dist.sample()
    action_log_probs = dist.log_prob(actions).sum(dim=-1)
  return values, actions, action_log_probs
def compute_loss(self, sample):
  ...
  # Function gets logits, values from model, creates distribution, then uses dist.entropy() to get entropy
  values, action_log_probs, dist_entropy = self.evaluate_actions(observations_batch, actions_batch)
  ...
  ratio = torch.exp(action_log_probs - old_action_log_probs) # pi / pi_old
  policy_loss = -torch.min(ratio * adv_targ, torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ).mean()
  value_loss = F.mse_loss(values, return_batch, reduction='none').mean() 
  loss = policy_loss + self.config.value_loss_weight * value_loss - self.config.entropy_loss_weight * dist_entropy
  loss = loss.mean()
  return loss, policy_loss_mean, value_loss_mean, torch.mean(dist_entropy), torch.mean(ratio)
```
# Assignment 4
```python
# Behavior Cloning Loss
pi_s = policy(states)
loss = F.mse_loss(pi_s, actions, reduction='mean')
return loss
# HG Dagger Training
for step in range(HG_STEPS):
  idx = np.random.choice(len(recov_data), BATCH_SIZE)
  hg_loss = compute_bc_loss(hg_model, recov_data.states[idx].to(device, recov_data.actions[idx].to(device)))
# DPO Loss
pi = policy(states)
with torch.no_grad():
  pi_ref = ref_policy(states)
pi_neg_error = ((pi - neg_actions)**2).sum(dim=-1)
pi_pos_error = ((pi - pos_actions)**2).sum(dim=-1)
ref_neg_error = ((pi_ref - neg_actions)**2).sum(dim=-1)
ref_pos_error = ((pi_ref - pos_actions)**2).sum(dim=-1)
loss = -F.logsigmoid((pi_neg_error - pi_pos_error) - (ref_neg_error - ref_pos_error)).mean()
```