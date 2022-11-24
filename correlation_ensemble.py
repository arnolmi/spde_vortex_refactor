import os
import solvers
import h5py

import numpy as np
import tqdm as tqdm
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import scipy
from tqdm import tqdm

#import cuda_api
#cuda = cuda_api.CudaAPI()

def monoExpZeroB(x,m,t):
    return m*np.exp(-x/t)

def detect_num_times(fd):
    num_keys = len(fd.keys())
    return num_keys

def extract_fields(fd):
    """
    Extracts the fields from an HDF5 file
    @fd the file descriptor
    """
    a1s = []
    a2s = []
    a3s = []
    a4s = []
    times = []
    first = list(fd.keys())[0]
    x_size, y_size = fd[first]['a1'][()].shape
    for key in fd.keys():
        a1 = fd[key]['a1'][()]
        a2 = fd[key]['a2'][()]
        a3 = fd[key]['a3'][()]
        a4 = fd[key]['a4'][()]
        if int(float(key)) % 3 == 0:
           continue
        times.append(float(key))

        a1s.append(a1)
        a2s.append(a2)
        a3s.append(a3)
        a4s.append(a4)
    times = np.array(times)
    a1s = np.array(a1s)
    a2s = np.array(a2s)
    a3s = np.array(a3s)
    a4s = np.array(a4s)

    # lets just go ahead and sort it all
    argidx = np.argsort(times)
    return a1s[argidx], a2s[argidx], a3s[argidx], a4s[argidx], times[argidx], x_size, y_size

def compute_correlation_length(a1, a3, x_size, y_size):
    corr = solvers.correlation_lengths(a1, a3, stride=5, method='FFT')
    distances = solvers.generate_grid_distances((x_size, y_size))

    return corr, distances

def get_correlation_dict(path, num_files = 10):
    mean_dict = {}
    count = 0
    for filename in tqdm(os.listdir(path)):
        if count == num_files:
            break
        
        corr_accumulator = None
        distances = None
        with h5py.File(os.path.join(path, filename), 'r') as fd:
            a1s, a2s, a3s, a4s, times, x_size, y_size = extract_fields(fd)
            corr, distance = compute_correlation_length(a1s, a3s, x_size, y_size)

            for t in range(0, len(times)):
                for x in range(0, x_size):
                    for y in range(0, y_size):
                        time_dict = mean_dict.get(t, {})
                        x_, y_ = time_dict.get(distance[x][y], (0, 1))
                        x_ += corr[t][x][y]
                        y_ += 1
                        #breakpoint()
                        time_dict[distance[x][y]] = (x_, y_)
                        mean_dict[t] = time_dict
        count += 1
    return mean_dict

def solve_for_correlation_scaling(mean_dict):
    times_test = []
    distances_test = []
    corr_test = []
    for time, time_dict in mean_dict.items():
        for distance, mean_tuple in time_dict.items():
            corr, num_samples = mean_tuple
            times_test.append(time)
            distances_test.append(distance)
            corr_test.append(corr / num_samples)
    result_dict = {"t": times_test, "corr": corr_test, "r": distances_test}
    df = pd.DataFrame(result_dict)
    df = df[df.r <= 125]
    df = df.groupby(['t', 'r']).mean().reset_index()    
    return df

def main(data_path, show_plot, file_name, num_files):
    parser = argparse.ArgumentParser(description="Calculate Correlation Length")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--show_plot", action='store_true')
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--num_files", required=True)

    corr_test, distance_test, times_test = [], [] , []
    mean_dict = get_correlation_dict(args.data_path, num_files)
    df = solve_for_correlation_scaling(mean_dict)

    times = df.t.unique()
    xi = np.zeros(times.shape)
    for i in times:
        res_df = df[df['t'] == i]
        correlations = res_df['corr'].to_numpy()
        distances = res_df['r'].to_numpy()
        p0 = (1,4.0)
        paramsB, _ = scipy.optimize.curve_fit(monoExpZeroB, distances, correlations, p0)
        xi[i] = paramsB[1]

    logt = np.log10(times)
    t = np.linspace(1.5, 2, 20)
    res, = plt.plot(logt, np.log10(xi), '.-')
    res, = plt.plot(t, 1.2*t-0.5)
    if args.show_plot:
        plt.show()
        
    save_location = os.path.join(data_path, '..', file_name)
    plt.savefig(save_location)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Correlation Length")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--show_plot", action='store_true')
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--num_files", required=True, type=int)

    args = parser.parse_args()
    num_files = args.num_files
    show_plot = args.show_plot
    file_name = args.file_name
    data_path = args.data_path
    main(data_path, show_plot, file_name, num_files)
