#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from user_taste import entp_reader
from music_space.initialize_music_space import initialize_music_space, get_all_artist_terms

def setup(mode = 1):


    ## setup python virtualenv

    if not os.path.exists("./env"):
        print("Setting up python virtual environment")
        os.system("python3 -m venv env")
        print("### Use the following cmd to enter virtual env:")
        print("source ./env/bin/activate")
        print("###")

    ## add the user_taste/data dir

    if not os.path.exists("./user_taste/data"):
        print("Creating ./user_taste/data directory")
        os.system("mkdir ./user_taste/data")


    ## download and process music space data

    os.chdir("./music_space")

    if not os.path.exists("./millionsongsubset.tar.gz"):
        print("# downloading million song subset ")
        os.system("wget http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz")
    if not os.path.exists("./MillionSongSubset"):
        print("# extracting ( this may take some time )")
        os.system("tar -xvf millionsongsubset.tar.gz")

    print("# initializing music space")

    if (
            (not os.path.exists(f"embeddings/csv/MSD_IDs_{mode}.csv") )
            or (not os.path.exists(f"embeddings/csv/MSD_songs_{mode}.csv") )
            or (not os.path.exists(f"embeddings/npy/MSD_features_{mode}.npy") )
            or (not os.path.exists(f"embeddings/npy/MSD_song_IDs_{mode}.npy") )
            ):

        if mode == 2:
            get_all_artist_terms()

        initialize_music_space(mode)

    if not os.path.exists("embeddings/MSD_features.npy"):
        os.system("cp embeddings/npy/MSD_features_1.npy embeddings/MSD_features.npy")

    if not os.path.exists("embeddings/MSD_song_IDs.npy"):
        os.system("cp embeddings/npy/MSD_song_IDs_1.npy embeddings/MSD_song_IDs.npy")

    os.chdir("..")


    ## download and process user taste dataset
    print("# user taste data")

    os.chdir("./user_taste/data")

    if not os.path.exists("train_triplets.txt"):
        print("Downloading dataset")

        os.system("wget http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip")
        os.system("unzip train_triplets.txt.zip")

    os.chdir("..")

    if ( 
            (not os.path.exists("./data/user_taste.npy"))
            ):
        print("Processing tastes in ./data/user_taste.npy")
        entp_reader.process("./data/train_triplets.txt")

    os.chdir("..")

    ## all done : )
    
    print("Setup done. Please run python3 main.py.")

def clean():
    os.remove("data/*")

if __name__=="__main__":

    if len(sys.argv)>1 and sys.argv[1] == "clean":
        clean()

    else:
        setup(mode=2)

