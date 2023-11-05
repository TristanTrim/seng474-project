#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from user_taste import entp_reader

def setup():

    did_something = False

    if not os.path.exists("data"):
        print("Creating ./data directory")
        os.system("mkdir data")
    if not os.path.exists("data/train_triplets.txt"):
        print("Downloading dataset")
        os.system("chmod 755 ./user_taste/get_entp.sh")
        os.system("./user_taste/get_entp.sh")
    if ( (not os.path.exists("data/sid.txt"))
            or (not os.path.exists("data/uid.txt")) ):
        print("Extracting sid and uid to sid.txt and uid.txt")
        did_something = True
        entp_reader.process("./data/train_triplets.txt", "./data/")
    
    if did_something:
        print("Setup done. Please run python3 main.py.")
    else:
        print("Already setup. Did nothing.")

def clean():
    os.remove("data/*")

if __name__=="__main__":

    if sys.argv[1] == "clean":
        clean()

    else:
        setup()

