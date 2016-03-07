import sys
import os
import os.path
import argparse

def count_spots(file_name):
    count = 0
    with open(file_name) as f:
        for line in f:
            if 'occupied="0"' in line:
                count += 1
    return count





