# Cross validation script for cnn.lua

import subprocess as proc
import numpy as np
import time

ITERS=100

print "{0}\t{1}\t{2}\t{3}".format(
       'Learning Rate', 'Batch Size', 'LR Decay Rate', 'LR Decay Every')

with open('crossval{0}.txt'.format(int(time.time())), 'w') as outfile:

    for i in xrange(ITERS):
        lr = np.power(10., np.random.uniform(-6, -1))
        bsize = np.random.choice(np.array([25, 50, 100]))
        lrd = np.random.choice(np.array([0.9, 0.95, 0.98, 0.99]))
        lrd_every = np.random.choice(np.array([1, 3, 5]))
        
        # TODO: add choices for conv_layers and fc_layers

        # Assemble call string
        callstr = "th cnn.lua -h5_file h5/pklot-small.hdf5 -num_epochs 1 -opt_method adam -print_every 0 -gpu 1 -learning_rate {0} -batch_size {1} -lr_decay_factor {2} -lr_decay_every {3}".format(lr, bsize, lrd, lrd_every)

        print callstr
        proc.call(callstr, stdout=outfile, shell=True)


        # Print results
        print "{0}\t{1}\t{2}\t{3}".format(lr, bsize, lrd, lrd_every)

