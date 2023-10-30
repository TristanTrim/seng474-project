############## INFO ABOUT THIS ALGORITHM #############

# Q-Learning is a Value-Based RL algorithm that optimizes the value function

# Q-Learning is an OFF-policy algorithm that learns based on the max Q-value
# of the next state-action pair

# Q-learning is generally more focused on maximizing the expected return and 
# is generally a more deterministic policy than SARSA

######################################################

def q_learning_loop(environment, training_rounds=100, alpha=0.1, gamma=0.5, epsilon=0.1):

    # Initialize Q-values
    q_table = {}
    for i in range(len(environment)):  # <- not sure what these will be based on yet
        q_table = {('sid', 'song value')} # <- need to replaces these with the metrics we decide (ie intial value function or could be arbitrary)

    # Select an inital song starting state
    state = random_state()


    # Start training loop
    for round in training_rounds:

        # Choose an action using an explore strategy (ie epsilon-greedy policy)
        action = explore_strategy(q_table[state])

        # Observe the next state and immediate reward
        next_state, reward = take_action(environment, state, action)

        # Update Q-value using Q-learning update -> Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        q_table[state][action] += alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[state][action])

        state = next_state

    return q_table