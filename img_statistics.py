
# coding: utf-8

# In[19]:

import h5py
import sys
import os
import os.path
import numpy as np
import argparse

from PIL import Image, ImageFilter
from count_spots import count_spots

parser = argparse.ArgumentParser(description="Compress PKLot dataset into HDF5 file")

parser.add_argument('--data_root', default=r'/home/jordan/Documents/PKLot/PKLotSegmented')
parser.add_argument('--count_spots', action='store_true')

params = vars(parser.parse_args())

WEATHER = ['Sunny', 'Cloudy', 'Rainy']
WEATHER_SET = set(WEATHER)

OCCUPIED = ['Empty', 'Occupied']
OCCUPIED_SET = set(OCCUPIED)

LOT = ['PUC', 'UFPR04', 'UFPR05']
LOT_SET = set(LOT)

# Parameters for statistics
stats = {'Sunny': 0, 'Cloudy': 0, 'Rainy': 0, 'Empty': 0, 'Occupied': 0}
stats_count_spots = {}

# User defined variables here
print "Counting images... (* = 1,000 added)"

image_count = 0
for path, dirs, files in os.walk(params['data_root']):

    # Get metadata of current folder
    tags = path.split(os.path.sep)

    # Check if we are iterating through spaces dirs or lots dirs
    if params['count_spots']:
        occupied = OCCUPIED[0]
        weather = tags[-2]
        lot = tags[-3]
    else:
        occupied = tags[-1]
    #     date = tags[-2]
        weather = tags[-3]
        lot = tags[-4]

    # Add image files
    for f in files:
        _, ext = os.path.splitext(f)
        if ext == '.jpg':

            #print (files)
            # Get Date and Time metadata
            date = f[:f.find('_')]
            year, month, day = tuple(date.split('-'))

            time = f[f.find('_') + 1: f.find('#')]
            hour, minute, second = tuple(time.split('_'))

            space = int(f[f.find('#') + 1:f.find('#') + 4])

            stats[occupied] += 1
            stats[weather] += 1

            if params['count_spots']:
                xmlpath = os.path.join(path, f[:-3] + 'xml')
                if not os.path.isfile(xmlpath):
                    print "Warning: could not find file '{0}'".format(xmlpath)
                    continue
                count = count_spots(path + '/' + f[:-3] + 'xml')
                #print "At {0}, empty {1}".format(os.path.join(path, f), count)
                stats_count_spots[count] = stats_count_spots.setdefault(count, 0) + 1


            image_count += 1
            if image_count % 1000 == 0:
                print "*",
                sys.stdout.flush()
            if image_count % 10000 == 0:
                print ""
                sys.stdout.flush()
            if image_count % 100000 == 0:
                print "\n"
                sys.stdout.flush()

# File statistics
print "\n\nImage statistics:\n"
sys.stdout.flush()

for i in [OCCUPIED, WEATHER]:
    for k in i:
        print "{0}: {1} ({2}%)".format(k, stats[k], 100 * stats[k] / float(image_count))
    print ""

if params['count_spots']:
    for k,v in stats_count_spots.iteritems():
       print "{0} Spots: {1} ({2}%)".format(k, v, 100 * v / float(image_count))

    
