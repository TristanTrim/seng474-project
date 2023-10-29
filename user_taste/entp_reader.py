#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def process(entp_file, output_dir):
    """
        Process ENTP text file, retrieve all distinct song IDs (sid.txt) and user IDs (uid.txt)
        
         Parameters:
             
            - input_file (str): The path to the input file ('train_triplets.txt') 
              containing the data to process.
              
            - output_dir (str): The directory where the processed data files 
              will be saved.
              
          Note:
              
            - train_triplets.txt stores entire ENTP dataset: rows containing user id, song id, and score, corresponding to the number of listens
              a user has listened to a song. There are 1M+ users, and 300,000+ song IDs from the Million Song Dataset,
              and a total of 40,000,000+ lines in ENTP. function is intended for use with train_triplets.txt only.
              
            - sid.txt is intended for use in populating the music space with songs assigned scores in ENTP.
              
            - Ensure that the 'output_dir' directory exists and is writable.
            
        """
    
    # retrieve unique user and song id's
    uid_set=set()
    sid_set=set()

    with open(entp_file,'r') as input_file:
        for line in input_file:
            uid,sid,_ = line.strip().split()
            uid_set.add(uid)
            sid_set.add(sid)
    

    # write to files
    with open(output_dir+"uid.txt","w") as output_file:
        for uid in uid_set:
            output_file.write(uid + '\n')
            
    with open(output_dir+"sid.txt","w") as output_file:
         for sid in sid_set:
             output_file.write(sid + '\n')
        
    return

    
    