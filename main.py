
from user_taste import user_taste
from game_engine.game_engine import GameEngine

def main():
    game_engine = GameEngine(tastes_set = user_taste)
    print( game_engine.run_game() )

if __name__=="__main__":
    main()

