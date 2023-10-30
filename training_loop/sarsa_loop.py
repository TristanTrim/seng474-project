############## INFO ABOUT THIS ALGORITHM #############

# SARSA (State-Action-Reward-State-Action) is a Value-Based RL algorithm 
# that optimizes the value function

# SARSA is an ON-policy algorithm and learns based on actions actually 
# taken by the agent

# SARSA considers the exploration strategy when updating its Q-values,
# so it is more inclined explore the environment than Q-learning

######################################################


def sarsa_loop(environment, num_episodes=100, alpha=0.1, gamma=0.5, epsilon=0.1):

    # Initialize Q-values
    q_table = {}
    for i in range(len(environment)):  # <- not sure what these will be based on yet

    # Select an inital song starting state
    state = random_state()

    # Start training loop
    for round in training_rounds:

        # Choose an action using an explore strategy (ie epsilon-greedy policy)
        action = explore_strategy(q_table[state])

        # Observe the next state and immediate reward
        next_state, reward = take_action(environment, state, action)

        # Choose the next action 
        next_action = explore_strategy(q_table[state])

        # Update Q-value using SARSA update -> Q(s, a) = Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])

        state = next_state
        action = next_action

    return q_table