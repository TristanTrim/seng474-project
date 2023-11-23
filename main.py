#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from agents.basic_mlp import Agent
from user_taste.user_taste import user_taste, MC_score_matrix
from music_space.music_space import music_space
from game_engine.game_engine import GameEngine
from training_loop.trainer import train

import sys

def main():
    agent = Agent()
    ms = music_space("./music_space/embeddings/npy/")
    ut = user_taste("./user_taste/data/")
    sm = MC_score_matrix('./user_taste/data/')
    game_engine = GameEngine(
                    agent = agent,
                    tastes_set = ut,
                    music_space = ms,
                    score_matrix= sm
                    )

    train( game_engine=game_engine,
            verbose=False )

def test_train():

    game_engine = GameEngine(tastes_set = user_taste, testmode=True)
    training_history = train(game_engine = game_engine, verbose = True)
   
    print(training_history) 

if __name__=="__main__":

    if sys.argv[1:2] == ["test_train"]:
        test_train()

    else:
        main()

