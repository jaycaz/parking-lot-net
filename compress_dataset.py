
# coding: utf-8

# In[19]:

import h5py
import sys
import os
import os.path
import numpy as np
import argparse
import time

from PIL import Image, ImageFilter
from count_spots import count_spots

parser = argparse.ArgumentParser(description="Compress PKLot dataset into HDF5 file")

parser.add_argument('--data_root', default=r'/home/jordan/Documents/PKLot')
parser.add_argument('--add_prop', type=float, default=1.0, help='Proportion of images to add to file')
parser.add_argument('--h5_name', default='pklot.hdf5', help='Name of HDF5 file to create')
parser.add_argument('--count_spots', action='store_true', help='If used, will assume the lot dset is used and will add empty space counts to h5 file')
parser.add_argument('--seed', type=int, default=-1)

params = vars(parser.parse_args())
#print params

if params['count_spots']:
    WIDTH = 256
    HEIGHT = 128

else:
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
stats_count_spots = {}

# Count all images before traversing directory
total_images = 0 
seg_root = os.path.join(params['data_root'], 'PKLotSegmented')
lot_root = os.path.join(params['data_root'], 'PKLot')

if params['count_spots']:
    data_root = lot_root
else:
    data_root = seg_root

print "Counting images..."
for path, dirs, files in os.walk(data_root):
    total_images += len([f for f in files if f.endswith('.jpg')])
print "Image count: ", total_images


if params['seed'] < 0:
    params['seed'] = int(time.time())

np.random.seed(params['seed'])
image_mask = np.random.binomial(1, params['add_prop'], size=(total_images,))
image_count = image_mask.sum()
print "Adding {0} images each with prob. {1} = {2} images".format(total_images, params['add_prop'], image_count)



sys.stdout.flush()


#fname = 'pklot.hdf5'
print "Creating HDF5 file ", params['h5_name']
print "Adding images... (* = 1,000 added)"

with h5py.File(params['h5_name'], 'w') as hf:

    data_dset = hf.create_dataset('data', (image_count, 3, HEIGHT, WIDTH), dtype='i')

    year_dset = hf.create_dataset('meta_year', (image_count,), dtype='i')
    month_dset = hf.create_dataset('meta_month', (image_count,), dtype='i')
    day_dset = hf.create_dataset('meta_day', (image_count,), dtype='i')

    hour_dset = hf.create_dataset('meta_hour', (image_count,), dtype='i')
    minute_dset = hf.create_dataset('meta_minute', (image_count,), dtype='i')
    second_dset = hf.create_dataset('meta_second', (image_count,), dtype='i')

    lot_dset = hf.create_dataset('meta_lot', (image_count,), dtype='i')
    space_dset = hf.create_dataset('meta_space', (image_count,), dtype='i')
    
    occupied_dset = hf.create_dataset('meta_occupied', (image_count,), dtype='i')
    weather_dset = hf.create_dataset('meta_weather', (image_count,), dtype='i')

    if params['count_spots']:
        count_dset = hf.create_dataset('meta_count_spots', (image_count,), dtype='i')

    step = 0 # Number of steps through directory
    img = 0 # Number of images actually added
    for path, dirs, files in os.walk(data_root):
        
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
            head, ext = os.path.splitext(f)
            if image_mask[step] == 1 and ext == '.jpg':
                print (f)
                # Get Date and Time metadata
                date = head[:head.find('_')]
                year, month, day = tuple(date.split('-'))

                time = head[head.find('_') + 1: head.find('#')]
                hour, minute, second = tuple(time.split('_'))

                space = int(head[head.find('#') + 1:head.find('#') + 4])

                # Extract image data, resize and add to file
                im = Image.open(os.path.join(path, f))
                im = im.resize((WIDTH, HEIGHT), Image.BICUBIC)

                imdata = np.asarray(im.getdata())
                imdata = imdata.reshape((3, im.height, im.width))
                im.close()
#                 print imdata.shape

                # Add all data to HDF5
                data_dset[img,:,:,:] = imdata

                year_dset[img] = int(year)
                month_dset[img] = int(month)
                day_dset[img] = int(day)

                hour_dset[img] = int(hour)
                minute_dset[img] = int(minute)
                second_dset[img] = int(second)

                #if not lot in LOT:
                  #print "ERROR: lot '{0}' not in list of lot names".format(lot)
                lot_dset[img] = LOT.index(lot) + 1
                space_dset[img] = space

                occupied_dset[img] = OCCUPIED.index(occupied) + 1
                stats[occupied] += 1
                weather_dset[img] = WEATHER.index(weather) + 1
                stats[weather] += 1

                if params['count_spots']:
                    xmlpath = os.path.join(path, head + '.xml')
                    if not os.path.isfile(xmlpath):
                        print "Warning: could not find file '{0}'".format(xmlpath)
                        continue
                    count = count_spots(xmlpath) + 1 # Adding one to count because Torch requires 1-indexed class labels
                    #print "At {0}, empty {1}".format(os.path.join(path, f), count)
                    count_dset[img] = count
                    stats_count_spots[count] = stats_count_spots.setdefault(count, 0) + 1

#                 if np.random.sample() < 1e-3:
#                     print month, day, hour, minute, lot, space, occupied, weather
#                     print month_dset[i], day_dset[i], hour_dset[i], minute_dset[i], lot_dset[i], space_dset[i], occupied_dset[i], weather_dset[i]
                img += 1
                if img % 1000 == 0:
                    print "*",
                    sys.stdout.flush()
                if img % 10000 == 0:
                    print ""
                    sys.stdout.flush()
                if img % 100000 == 0:
                    print "\n"
                    sys.stdout.flush()

            # Increment image counters
            if ext == '.jpg':
                step += 1
            if img >= image_count:
                break
        if img >= image_count:
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

    if params['count_spots']:
        for k,v in stats_count_spots.iteritems():
           print "{0} Spots: {1} ({2}%)".format(k, v, 100 * v / float(image_count))

    
    
#     path_tags = set(params['data_root'].split(os.path.sep))
#     print list(path_tags)
#     weather = path_tags.intersection(WEATHER).pop()
#     print weather
#     space = path_tags.intersection(SPACE).pop()
#     print space
#     break
#     print path, dirs






