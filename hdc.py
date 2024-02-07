import math                         
import random                     
import numpy as np                
import numpy.linalg as lin        
import scipy.special as ss
import pickle
import multiprocessing
import time
import pandas as pd
import sys
import csv

def distance(A,B):
    return sum(np.logical_and(A,B))

def perm(A,N):
    return np.roll(A,N)

def u_gen_rand_hv(D,d):
    # Sanity checker
    if (D % 2):
        print("Error - D can't be an odd number")
        return 0
    # generate
    chosen = random.sample(range(D), k =int(D*d))
    rand_hv = np.array([1 if x in chosen else 0 for x in range(D)])
    return rand_hv

def create_item_mem(N,D,d):
    keys = range(N)
    seed = u_gen_rand_hv(D,d) #Generate List of random 1 and 0 with probability d
    tracker = pd.Series(np.copy(seed)) #Tracks already flipped bits
    bit_step = int(np.sum(seed)/(len(keys)-1))
    hvs = [seed]

    for i in range(1,len(keys)):
        next_hv = np.copy(hvs[i-1])

        # TURN OFF K bits
        turnoff_index = random.sample(list(tracker[tracker==1].index), bit_step)
        tracker[turnoff_index]=-1 #Update to already flipped
        next_hv[turnoff_index]=0 #flip to 0

        # TURN ON K bits
        turnon_index = random.sample(list(tracker[tracker==0].index), bit_step)
        tracker[turnon_index]=-1 #Update to already flipped
        next_hv[turnon_index]=1 #Flip to 1

        hvs.append(next_hv)
       
    return dict(zip(keys,hvs))

def hdc_encode(voice,voice_im,D,d,Q=10, t1 = 0, remove_list = []): #d = density, Q - quqntization steps, t1 first threshold
    voice = [ int(math.floor(x*(Q/2))+(Q/2)) if x<1 else int(math.floor((x-0.0001)*(Q/2))+(Q/2)) for x in voice]
    feature_hv_list = [voice_im[x] for x in voice]
    feature_hv_list = np.array([perm(feature_hv_list[x],x) for x in range(len(feature_hv_list))])
    
    threshold = t1 #first threshold
    bundle = np.sum(feature_hv_list,axis=0)
    
    if remove_list:
        np.delete(bundle, remove_list)
        
    #for threshold in range(1000):
    bndl = np.where(bundle>threshold,1,0)
    #    print(threshold,sum(bndl))
    return bndl

def hdc_enc(args):
    return hdc_encode(*args)

def dimension_sparsifier(non_bin_reg, dimensionality, sparsity, remove_list):
    remove_num = int(sparsity * dimensionality)
    #print(remove_num)
    max_list = non_bin_reg[0]
    min_list = non_bin_reg[0]
    
    for i in range(1,len(non_bin_reg)):
        max_list = np.maximum(max_list, non_bin_reg[i])
        min_list = np.where(np.logical_or(min_list>=non_bin_reg[i], min_list==0), non_bin_reg[i], min_list)

    var_list = np.subtract(max_list, min_list)
    var_list = sorted(enumerate(var_list), key=lambda x:x[1])
    idx, value = map(list, zip(*var_list))
    remove_list = idx[0:remove_num]

    for i in range(len(non_bin_reg)):
        np.delete(non_bin_reg[i], remove_list)
    return non_bin_reg, remove_list

def similarity_search(voice,voice_im,voice_am,D,d,Q,t1, remove_list):
    # insert nice code here
    start = time.time()
    sim_score = 0
    sim_letter = '0'
    test_hv = hdc_encode(voice,voice_im,D,d,Q,t1,remove_list)
    encoding_time = time.time() -start
    start = time.time()
    for each in voice_am.items():
    #print(each)
        similarity = np.sum(np.logical_and(test_hv,each[1]))
        if (similarity>sim_score):
            sim_score,sim_letter = similarity, each[0]
  
    return (sim_letter,sim_score,encoding_time, time.time()-start)

def search(args):
    return similarity_search(*args)
