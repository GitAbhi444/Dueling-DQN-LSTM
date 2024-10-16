#-----------------------------------------------------------------------------
# Dueling DQN model with LSTM
class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_pairs, lstm_units):
        super(DuelingDQN, self).__init__()
        self.lstm = nn.LSTM(input_size=state_shape[0], hidden_size=lstm_units, batch_first=True)
        self.fc1 = nn.Linear(lstm_units, 64)
        self.value_fc = nn.Linear(64, 64)
        self.advantage_fc = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, len(action_pairs))
        
    def forward(self, state):
        state = state.float()  # Ensure input is float
        lstm_out, _ = self.lstm(state)
        x = torch.relu(self.fc1(lstm_out[:, -1, :]))  # Taking the last LSTM output (sequence length dimension)
        
        value_out = torch.relu(self.value_fc(x))
        value = self.value(value_out)
        
        advantage_out = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage_out)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))  # Dueling architecture equation
        return q_values
# DQN Agent
class DQNAgent:
    def __init__(self, state_shape, action_pairs, alpha, gamma, lstm_units, buffer_size, sample_size):
        self.state_shape = state_shape
        self.action_pairs = action_pairs
        self.num_actions = len(action_pairs)
        self.alpha = alpha
        self.gamma = gamma
        self.lstm_units = lstm_units
        self.replay_buffer = deque(maxlen=buffer_size)
        self.sample_size = sample_size

        self.primary_net = DuelingDQN(self.state_shape, self.action_pairs, self.lstm_units)
        self.target_net = DuelingDQN(self.state_shape, self.action_pairs, self.lstm_units)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.primary_net.parameters(), lr=self.alpha)
        self.mse_loss = nn.MSELoss()

        # Move networks to the appropriate device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.primary_net.to(self.device)
        self.target_net.to(self.device)

        # Pass dummy data to both networks to initialize their weights
        dummy_input = torch.zeros((1, 1, self.state_shape[0])).to(self.device)
        self.primary_net(dummy_input)  # Ensure primary_net is initialized
        self.target_net(dummy_input)   # Ensure target_net is initialized
        
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.primary_net.state_dict())

    def reduce_epsilon(self, epsilon, epsilon_decay, epsilon_end):
        return max(epsilon * epsilon_decay, epsilon_end)

    def choose_action(self, state, epsilon):
        state = torch.tensor(np.array(state).reshape((1, 1, self.state_shape[0]))).float().to(self.device)
        if random.uniform(0, 1) < epsilon:
            action_index = random.choice(range(self.num_actions))  # Explore action space
        else:
            q_values = self.primary_net(state)
            action_index = q_values.argmax().item()  # Exploit learned values
        action = self.action_pairs[action_index]  # Access the corresponding action from the list
        return action, action_index  # Return both the action and its index

    def store_experience(self, state, action_index, reward, next_state):
        experience = (state, action_index, reward, next_state)
        self.replay_buffer.append(experience)  # Store experience in buffer

    def sample_experiences(self):
        buffer_size = len(self.replay_buffer)
        if buffer_size >= self.sample_size:
            indices = random.sample(range(buffer_size), self.sample_size)
            return [self.replay_buffer[i] for i in indices]
        return []

    def learn(self, state, action_index, reward, next_state):
        self.store_experience(state, action_index, reward, next_state)  # Store experience in buffer
        loss = self.learn_from_batch()  # Update Q-values from a batch      
        return loss

    def learn_from_batch(self):
        sample_batch = self.sample_experiences()
        if not sample_batch:
            return 0.0  # Exit if there's no sufficient experience
        
        states, action_indices, rewards, next_states = zip(*sample_batch)

        # Prepare state and next_state tensors for LSTM input
        states = torch.tensor(np.array(states)).float().reshape(self.sample_size, 1, self.state_shape[0]).to(self.device)
        next_states = torch.tensor(np.array(next_states)).float().reshape(self.sample_size, 1, self.state_shape[0]).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        action_indices = torch.tensor(action_indices).long().to(self.device)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]  
            target_q_values = rewards + self.gamma * next_q_values  
    
        q_values = self.primary_net(states)
        q_values = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)    
        self.optimizer.zero_grad()  # Clear previous gradients
        loss = self.mse_loss(q_values, target_q_values)  # Calculate loss
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Return the loss for tracking
#----------------------------------------------------------------------------------------
