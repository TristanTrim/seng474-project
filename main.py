#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from user_taste.user_taste import user_taste
from game_engine.game_engine import GameEngine
from training_loop.trainer import train

import sys

def main():
    game_engine = GameEngine(tastes_set = user_taste)
    print( game_engine.run_game() )

def test_train():

    game_engine = GameEngine(tastes_set = user_taste, testmode=True)
    training_history = train(game_engine = game_engine, verbose = True)
   
    print(training_history) 

if __name__=="__main__":

    if sys.argv[1:2] == ["test_train"]:
        test_train()

    else:
        main()

