import time
import os
import random

import scipy
import pandas as pd
import numpy as np
import h5py
import argparse

from tqdm import tqdm
from multiprocessing import Process, Manager, cpu_count, Pool

import vortex
import solvers


def find_vortexes_ts(data_fd, x_size, y_size):
    vortex_locations = []
    angles = []
    times = []
    energy = []
    for time_t, series in data_fd.items():
        a1 = series['a1'][()]
        a2 = series['a2'][()]
        a3 = series['a3'][()]
        a4 = series['a4'][()]

        en = series.attrs['en']
        en2 = series.attrs['en2']
        en3 = series.attrs['en3']
        ke = series.attrs['ke']
        tot = series.attrs['tot']

        angle = np.arctan2(a1, a3)
        pos, neg = vortex.find_vortexes(angle, radius=5, boundary=(x_size, y_size))
        vortex_locations.append((pos, neg))
        angles.append(angle)
        times.append(time_t)
        energy.append((en, en2, en3, ke, tot))

    return vortex_locations, angles, times, energy

def dir_is_empty(path):
    directory = os.listdir(path)
    return len(directory) == 0

def save_file(target_data_path, vortex_locations, angles, times, energy, x_size, y_size, filename):
    file_hash = random.getrandbits(128)
    write_fd = h5py.File(os.path.join(target_data_path, "underdamped_{}_angles.hdf5".format(file_hash)), 'a')
    for angle, time, energy, locations in zip(angles, times, energy, vortex_locations):
        try:
            g = write_fd.create_group(str(time))
        except:
            breakpoint()
        data = g.create_dataset('angles', (x_size, y_size), dtype='float64', chunks=(x_size, y_size), compression='gzip')
        data[:,:] = angle

        pos, neg = locations
        g['num_vortexes'] = pos + neg

        g.attrs['en2'] = energy[1]
        g.attrs['en3'] = energy[2]
        g.attrs['ke'] = energy[3]
        g.attrs['tot'] = energy[4]
    write_fd.attrs['dims'] = (x_size, y_size)
    write_fd.attrs['from_file'] = filename

def process_file(args):
    source_data_path, target_data_path, filename = args
    fd = h5py.File(os.path.join(source_data_path, filename), 'r')

    #extract variables
    print("Loading filename {}".format(filename))
    fd = h5py.File(os.path.join(source_data_path, filename), 'r')

    # check the dimensions
    num_times = len(fd.keys())
    first = list(fd.keys())[0]
    x_size, y_size = fd[first]['a1'][()].shape
    shape = (num_times, x_size, y_size)

    en = en2 = en3 = ke = tot = np.zeros((len(fd.keys())))

    vortex_locations, angles, times, energy = find_vortexes_ts(fd, x_size, y_size)
    save_file(target_data_path, vortex_locations, angles, times, energy, x_size, y_size, filename)

def main(source_data_path='./temp_results', target_data_path = 'results', num_threads = 10):
    manager = Manager()
    write_lock = manager.Lock()
    data_queue = manager.Queue()

    for filename in os.listdir(source_data_path):
        data_queue.put(filename)

    files = [filename for filename in os.listdir(source_data_path)]
    assert dir_is_empty(target_data_path), "Target Directory {} is not empty".format(target_data_path)
    
    source_data_paths = [source_data_path for filename in files]
    target_data_paths = [target_data_path for filename in files]
    p = Pool(num_threads)
    p.map(process_file, zip(source_data_paths, target_data_paths, files))


parser = argparse.ArgumentParser(description="Complex Fields into Angles")
parser.add_argument('--source_data_path', help='Location of the source data', required=True)
parser.add_argument('--target_data_path', help='location of the results', required=True)
parser.add_argument('--num_threads', help='number of threads', type=int, required=True)
args = parser.parse_args()

main(args.source_data_path, args.target_data_path, args.num_threads)
