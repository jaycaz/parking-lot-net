# Cross validation script for cnn.lua

import subprocess as proc
import numpy as np
import time

ITERS=100

with open('generalize{0}.txt'.format(int(time.time())), 'w') as outfile:

    weather = ['sunny', 'rainy', 'cloudy']
    lots = ['PUC', 'UFPR04', 'UFPR05']

    for cond in [lots]: #[weather, lots]:
        for c1 in cond:
            for c2 in cond:
                # Assemble call string
                callstr = "th cnn.lua -h5_file h5/pklot-small.hdf5 -num_epochs 1 -print_every 2000 -gpu 1 -train_set {0} -test_set {1}".format(c1, c2)
    
                print callstr
                proc.call(callstr, stdout=outfile, shell=True)
    
    
                # Print results
                print "*{0}\t{1}".format(c1, c2)

