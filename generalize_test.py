# Cross validation script for cnn.lua

import subprocess as proc
import numpy as np
import time

ITERS=1

with open('generalize{0}.txt'.format(int(time.time())), 'w') as outfile:

    all_conds = {'weather': ['sunny', 'rainy', 'cloudy'], 'lot': ['PUC', 'UFPR04', 'UFPR05']}

    for name, conds in all_conds.iteritems():
        for cond in conds:
            for i in xrange(ITERS):
                # Assemble call string
                callstr = "th cnn.lua -h5_file h5/pklot-small.hdf5 -num_epochs 1 -print_every 0 -batch_norm 1 -gpu 1 -{0}_train {1}".format(name, cond)
    
                print callstr
                proc.call(callstr, stdout=outfile, shell=True)
    

