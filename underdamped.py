from __future__ import print_function

import random
from pandas import DataFrame
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from tqdm import tqdm

import time
import solvers
import h5py
import time

import argparse

# This must be the first statement before other statements.
# You may only put a quoted or triple quoted string, 
# Python comments, other future statements, or blank lines before the __future__ line.

try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__

def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature 
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    #__builtin__.print('My overridden print() function!')
    return __builtin__.print(*args, **kwargs)


grid_size = 256

# Don't actually change anything in the config, it mostly isn't used.
valid_config = {
    'name': "GL Theory Simulation",
    'dimensions': 2,
    'fields': 4,
    'ranges': (grid_size, grid_size),
    'observables': [lambda f: f, lambda f: f],
    'seed': 5,
    'steps': 400,
    'noises': 4,
    'da': [lambda f: f, lambda f: f]
}

def main(iteration, run_time, step_size, grid_size, data_path):
    valid_config['ranges'] = (grid_size, grid_size)
    config = solvers.SimulationConfig(valid_config)
    
    prefix_hash = random.getrandbits(128)
    for parameter in np.arange(-1, -2, -1):
        parameter = -1

        overdamped = solvers.OverdampedPhi4(parameter)
        underdamped = solvers.UnderdampedPhi4(parameter, overdamped)

        #rk4 = solvers.RK4(config,  underdamped)
        rk4 = solvers.VelocityVerlet(config, underdamped)

        dt = step_size
        string = "underdamped_{}_ud".format(grid_size)
        fd = h5py.File('{}/candidate_{}-{}-{}-{}-gs1-dt-{}.hdf5'.format(data_path, string, prefix_hash, str(parameter), int(iteration), dt), 'w')
        start_time = time.time()
        overdamped = True
        run_once = False
        while True:
            en, en2, en3, ke, tot, real_time, steps = rk4.step(dt)
            if rk4._time < 50:
                continue

            if rk4._time >= 50 and rk4._time < 100:
                rk4.clear_use_overdamped()

            if rk4._time >= 100 and rk4._time <= 150:
                rk4.clear_thermalize()
            elif rk4._time >= 150 and run_once == False:
                run_once = True
                pass
            if  rk4._steps % 1000 == 0:
                print(real_time)
                a,b,c = rk4.get_engine_parameters()
                print("therm: {} engine: {} parameter: {}\n".format(a,b,c))
                g = fd.create_group("{}".format(real_time ))
                en = solvers.get_real_numpy_array(en)
                en2 = solvers.get_real_numpy_array(en2)
                en3 = solvers.get_real_numpy_array(en3)
                ke = solvers.get_real_numpy_array(ke)
                tot = solvers.get_real_numpy_array(tot)
                g.attrs['en'] = en
                g.attrs['en2'] = en2
                g.attrs['en3'] = en3
                g.attrs['ke'] = ke
                g.attrs['tot'] = tot
                g.attrs['steps'] = steps
                g.attrs['real_time'] = real_time
                a1 = solvers.get_real_numpy_array(rk4._field_matrix[0])
                a2 = solvers.get_real_numpy_array(rk4._field_matrix[1])
                print(a1)
                a3 = solvers.get_real_numpy_array(rk4._field_matrix[2])
                a4 = solvers.get_real_numpy_array(rk4._field_matrix[3])
                field_data = g.create_dataset('a1', (grid_size,grid_size), dtype='float64', chunks=(grid_size,grid_size), compression='gzip')
                field_data[:,:] = a1

                field_data = g.create_dataset('a2', (grid_size,grid_size), dtype='float64', chunks=(grid_size,grid_size), compression='gzip')
                field_data[:,:] = a2

                field_data = g.create_dataset('a3', (grid_size,grid_size), dtype='float64', chunks=(grid_size,grid_size), compression='gzip')
                field_data[:,:] = a3

                field_data = g.create_dataset('a4', (grid_size,grid_size), dtype='float64', chunks=(grid_size,grid_size), compression='gzip')
                field_data[:,:] = a4

                print("time: {}".format(real_time))
                print("runtime: {}".format(time.time() - start_time))
                start_time = time.time()
            if real_time >= run_time:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Runs and Energy Conserving Underdamped Simulation")
    parser.add_argument('-c', action='store_true')
    parser.add_argument('--run_for_realtime', type=int, required=True)
    parser.add_argument('--step_size', type=float, required=True)
    parser.add_argument('--grid_size', type=int, required=True)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()
    
    run_time = args.run_for_realtime
    step_size = args.step_size
    grid_size = args.grid_size
    data_path = args.data_path
    
    for x in tqdm(range(0,5)):
        print("Iteration {}".format(x))
        main(x, run_time, step_size, grid_size, data_path)
