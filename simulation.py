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
import os
import hdc

if __name__ == '__main__':

    #Set Parameters
    D = 8192 #Dimensionality
    Q = 16   #Quantization Levels
    d = 0.1  #Density
    remove_list = []
    #Import ISOLET Dataset 
    with open('isolet.pkl', 'rb') as f:
        isolet = pickle.load(f)
    trainData, trainLabels, testData, testLabels = isolet
    trainData = np.array(trainData)
    testData = np.array(testData)

    Accuracies = []
    Encoded_densities = []
    AM_densities = []
    
    for encoding_threshold in range(50,121,5):
        
        accuracy_line = []
        encoded_density_line = []
        am_density_line = []
        
        training_range = range(100,101,5) # Run for steps of 5 until 150 (0,1,5) (5,6,5) ... (150,151,5)
        for training_threshold in training_range:
            
            confusion_mtx = np.zeros((26,26))
            print(encoding_threshold,training_threshold)
            
            trials = 10
            
            for trial in range(trials):
                keys = range(26)
                voice_am = dict()
                cores = multiprocessing.cpu_count()
                
                try:
                    #Read item memory and associative from file if it already exists
                    voice_im = np.array(pd.read_csv(f'im_{encoding_threshold}_{training_threshold}_{trial}.csv',header = None).values.tolist())
                    voice_im = dict(zip(range(26),voice_im))
                    voice_am = np.array(pd.read_csv(f"am_{encoding_threshold}_{training_threshold}_{trial}.csv",header = None).values.tolist())
                    voice_am = dict(zip(range(26),voice_am))
                    
                except:
                    #Create item memory    
                    voice_im = hdc.create_item_mem(Q,D,d)
                    im = pd.DataFrame(voice_im.values())
                    im.to_csv(f'im_{encoding_threshold}_{training_threshold}_{trial}.csv', index=False, header=False)
                    #print("Trial:",trial+1)

                    non_bin_reg = np.zeros((26,D))
                    
                    #print('threads used:',cores)
                    #print('Training...')

                    # Create the process pool
                    enc_start = time.time()
                    with multiprocessing.Pool(cores) as pool:
                        args = [(trainData[i],voice_im,D,d,Q,encoding_threshold) for i in range(len(trainLabels))]
                        results = pool.starmap(hdc.hdc_encode,args)
                    #print(results)
                    #encoded_csv = pd.DataFrame(np.concatenate(results))
                    #encoded_csv.to_csv(f'encoded_{int(min(training_range))}_{int(max(training_range))}_{trial}.csv', index=False, header=False)
                    #print('encode_time',time.time()-enc_start)
                    encoded_density = sum([sum(results[i]) for i in range(len(trainData))])/(D*len(trainData))
                    encoded_density_line.append(encoded_density)
                    print('Encoded HV Density:',encoded_density)
                    
                    for i in range(len(trainLabels)):
                        non_bin_reg[trainLabels[i]] = np.add(non_bin_reg[trainLabels[i]],results[i])
                    #non_bin_reg, remove_list = dimension_sparsifier(non_bin_reg, D, d, remove_list)
            
            
                
                    for i in range(len(keys)):
                        voice_am[keys[i]] = np.where(non_bin_reg[i]>training_threshold,1,0)
                        #print(i,sum(voice_am[keys[i]]))
                    
                    am_csv = pd.DataFrame.from_dict(voice_am, orient='index')
                    am_csv.to_csv(f'am_{encoding_threshold}_{training_threshold}_{trial}.csv', index=False, header=False)
                    
                am_density = sum([sum(list(voice_am.items())[i][1]) for i in range(len(keys))])/(D*len(keys))
                print('AM HV Density:',am_density)
                am_density_line.append(am_density)
                
                # Iterate through all elements in the clean_letters set
                #print('Testing')
                test_data = testData
                correct_values = testLabels
                print_flag = False
                score = 0
                test_len = len(test_data)
                #cores = multiprocessing.cpu_count()

                with multiprocessing.Pool(cores) as pool:
                    arg = [(test_data[i],voice_im,voice_am,D,d,Q,encoding_threshold,remove_list) for i in range(test_len)]
                    result = pool.map(hdc.search,arg)

                #print(len(result))
                enc_t = 0
                s_t = 0
                for i in range(test_len):
                    sim_letter, sim_score = result[i][0],result[i][1]
                    enc_t += result[i][2]
                    s_t += result[i][3]
                    if sim_letter == correct_values[i]:
                        score += 1
                        #confusion_mtx[correct_values[i]][sim_letter] += 1
                        if(print_flag):
                            print("CORRECT prediction! sim_letter: ", sim_letter, " sim_score: ", str(sim_score))
                    else:
                        #confusion_mtx[correct_values[i]][sim_letter] += 1
                        if(print_flag):
                            print("WRONG prediction! sim_letter: " , sim_letter, " sim_score: ", str(sim_score))
                #print(enc_t,s_t)
                #print("Final accuracy is: %f" % ((score*100/test_len)))
                accuracy = (score*100/test_len)
                print('Accuracy:',accuracy)
                accuracy_line.append(accuracy)
            '''
            for i in range(26):
                for j in range(26):
                    confusion_mtx[i][j] = round(confusion_mtx[i][j]/(1*trials),3)
            confusion_mtx_csv = pd.DataFrame(confusion_mtx)
            confusion_mtx_csv.to_csv(f'confusion_mtx_{encoding_threshold}_{training_threshold}_{trials}.csv', index=False, header=False)
            
            '''
            
            
            #plt.rcParams['figure.figsize'] = [8,6]
            #plt.rcParams['figure.dpi'] = 500
            #alphabet = [chr(i+65) for i in range(26)]
            #sns.palplot(sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0))
            #cmap = sns.cubehelix_palette(n_colors=50, start=1, rot=0, light=0.85, dark=0)
            #ax = sns.heatmap(confusion_mtx,xticklabels=alphabet,yticklabels=alphabet,square=True,cmap=cmap,linewidths=1,linecolor='white')
            #plt.ylabel('Actual')
            #plt.xlabel('Predicted')
            #plt.show()
        
        Accuracies.append(accuracy_line)
        Encoded_densities.append(encoded_density_line)
        AM_densities.append(am_density_line)
    #Save data to csv
    Accuracies.append([int(min(training_range)),int(max(training_range)),trials])
    Encoded_densities.append([int(min(training_range)),int(max(training_range)),trials])
    AM_densities.append([int(min(training_range)),int(max(training_range)),trials])
    Accuracy_csv = pd.DataFrame(Accuracies)
    Encoded_densities_csv = pd.DataFrame(Encoded_densities)
    AM_densities_csv = pd.DataFrame(AM_densities)
    Accuracy_csv.to_csv(f'Accuracy_{int(min(training_range))}_{int(max(training_range))}_{trials}.csv', index=False, header=False)
    Encoded_densities_csv.to_csv(f'Encoded_densities_{int(min(training_range))}_{int(max(training_range))}_{trials}.csv', index=False, header=False)
    AM_densities_csv.to_csv(f'AM_densities_{int(min(training_range))}_{int(max(training_range))}_{trials}.csv', index=False, header=False)
