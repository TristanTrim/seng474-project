#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def calc_returns( round_history, gamma):

    round_returns = []
    rewwwardddd = 0

    for i in range( len(round_history)-1, -1, -1 ):

        this_rounds_reward = round_history[i][1]
        speculative_contribution_to_future_reward = rewwwardddd*gamma

        rewwwardddd = (
            this_rounds_reward
            + speculative_contribution_to_future_reward )

        round_returns = (
            [ round_history[i] + ( rewwwardddd ,) ]
                            + round_returns
            )

    return( round_returns )
        

def train(
            alpha = 1e-3,
            gamma = 1-1e-1,
            game_engine = None,
            game_rounds = 10,

            verbose = False
        ):
   
    training_history = [] 

    for round_number in range(1,game_rounds+1):
        
        round_history = game_engine.run_game()
        if verbose:
            print(f"=== round {round_number} results: ===")
            print(f"{round_history}\n")

        round_returns = calc_returns(round_history, gamma)
        if verbose: print(f"=== returns: ===\n{round_returns}\n\n\n")
        
        loss = game_engine.agent.update_weights( round_returns )

        training_history += [ loss ]

    return( training_history )

