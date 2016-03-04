
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
parser.add_argument('--add_prop', type=float, default=1.0, help='Proportion of images to add to file')
parser.add_argument('--h5_name', default='pklot.hdf5', help='Name of HDF5 file to create')
parser.add_argument('--count_spots', action='store_true')

params = vars(parser.parse_args())
#print params

WIDTH = 48
HEIGHT = 64

WEATHER = ['Sunny', 'Cloudy', 'Rainy']
WEATHER_SET = set(WEATHER)

OCCUPIED = ['Empty', 'Occupied']
OCCUPIED_SET = set(OCCUPIED)

LOT = ['PUC', 'UFPR04', 'UFPR05']
LOT_SET = set(LOT)

# Parameters for statistics
stats = {'Sunny': 0, 'Cloudy': 0, 'Rainy': 0, 'Empty': 0, 'Occupied': 0}

# User defined variables here
#root = r'C:\Users\jacaz_000\Downloads\PKLot\PKLotSegmented'
#root = r'/Users/martina/Documents/Uni/USA/Stanford/2.Quarter/CNN/Finalproject/PKLot/PKLotSegmented2'
#root = r'/home/jordan/Documents/PKLot/PKLotSegmented'
#add_prob = 1.0 # Change this to add fewer files

total_images = 0 
print "Counting images..."
for path, dirs, files in os.walk(params['data_root']):
    total_images += len(files)
print "Image count: ", total_images


image_mask = np.random.binomial(1, params['add_prop'], size=(total_images,))
image_count = image_mask.sum()
print "Adding {0} images each with prob. {1} = {2} images".format(total_images, params['add_prop'], image_count)



sys.stdout.flush()


#fname = 'pklot.hdf5'
print "Creating HDF5 file ", params['h5_name']
print "Adding images... (* = 1,000 added)"

with h5py.File(params['h5_name'], 'w') as hf:

    data_dset = hf.create_dataset('data', (image_count, 3, HEIGHT, WIDTH), dtype='i')

    # year_dset = hf.create_dataset('meta_year', (image_count,), dtype='i')
    month_dset = hf.create_dataset('meta_month', (image_count,), dtype='i')
    day_dset = hf.create_dataset('meta_day', (image_count,), dtype='i')

    hour_dset = hf.create_dataset('meta_hour', (image_count,), dtype='i')
    minute_dset = hf.create_dataset('meta_minute', (image_count,), dtype='i')
    # second_dset = hf.create_dataset('meta_second', (image_count,), dtype='i')

    lot_dset = hf.create_dataset('meta_lot', (image_count,), dtype='i')
    space_dset = hf.create_dataset('meta_space', (image_count,), dtype='i')
    
    occupied_dset = hf.create_dataset('meta_occupied', (image_count,), dtype='i')
    weather_dset = hf.create_dataset('meta_weather', (image_count,), dtype='i')

    if params['count_spots']:
        count_dset = hf.create_dataset('meta_count_spots', (image_count,), dtype='i')

    i = 0
    for path, dirs, files in os.walk(params['data_root']):
        #if len(files) == 0:
        #    continue

        # Get metadata of current folder
        tags = path.split(os.path.sep)

        # Check if we are iterating through spaces dirs or lots dirs
        if not tags[-1] in OCCUPIED:
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

              # Extract image data, resize and add to file
              im = Image.open(os.path.join(path, f))
              im = im.resize((WIDTH, HEIGHT), Image.BICUBIC)

              imdata = np.asarray(im.getdata())
              imdata = imdata.reshape((3, im.height, im.width))
              im.close()
#               print imdata.shape

              # Add all data to HDF5
              data_dset[i,:,:,:] = imdata

              month_dset[i] = int(month)
              day_dset[i] = int(day)

              hour_dset[i] = int(hour)
              minute_dset[i] = int(minute)

              #if not lot in LOT:
                #print "ERROR: lot '{0}' not in list of lot names".format(lot)
              lot_dset[i] = LOT.index(lot) + 1
              space_dset[i] = space

              occupied_dset[i] = OCCUPIED.index(occupied) + 1
              stats[occupied] += 1
              weather_dset[i] = WEATHER.index(weather) + 1
              stats[weather] += 1

              if params['count_spots']:
                  count = count_spots(path + '/' + f[:-3] + 'xml')
                  #print "At {0}, empty {1}".format(os.path.join(path, f), count)
                  count_dset[i] = count

#               if np.random.sample() < 1e-3:
#                   print month, day, hour, minute, lot, space, occupied, weather
#                   print month_dset[i], day_dset[i], hour_dset[i], minute_dset[i], lot_dset[i], space_dset[i], occupied_dset[i], weather_dset[i]

              i += 1
              if i % 1000 == 0:
                  print "*",
                  sys.stdout.flush()
              if i % 10000 == 0:
                  print ""
                  sys.stdout.flush()
              if i % 100000 == 0:
                  print "\n"
                  sys.stdout.flush()
#             if image_mask[i] == 0:
#                 continue
            if i >= image_count:
                break
        if i >= image_count:
            break

    hf.close()

    # File statistics
    print "File successfully created!"
    print "Some statistics...\n"
    sys.stdout.flush()

    for i in [OCCUPIED, WEATHER]:
        for k in i:
            print "{0}: {1} ({2}%)".format(k, stats[k], 100 * stats[k] / float(image_count))
        print ""

    
    
#     path_tags = set(params['data_root'].split(os.path.sep))
#     print list(path_tags)
#     weather = path_tags.intersection(WEATHER).pop()
#     print weather
#     space = path_tags.intersection(SPACE).pop()
#     print space
#     break
#     print path, dirs






