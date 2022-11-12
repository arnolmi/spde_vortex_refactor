import matplotlib.pyplot as plt
import h5py
import os
import re
import numpy as np
#file_regex = '6underdamped_[0-9]+_angles\.hdf5'
times = []
vortex_counts = []
for filename in os.listdir('./'):
    #breakpoint()
    if not filename.startswith('underdamped'):
        break
    fd = h5py.File(filename, 'r')
    for t, v in fd.items():
        vortex_locations = v['num_vortexes']
        num_vortexes = len(vortex_locations)
        vortex_counts.append(num_vortexes)
        times.append(round(float(t), 2))


avg = {}
for x, y in zip(times, vortex_counts):
    curr, count = avg.get(x, (0,0))
    avg[x] = curr + y, count + 1

times = []
vortex_counts = []
for k, v in avg.items():
    times.append(k)
    curr, count = v
    vortex_counts.append(curr / count)

idx = np.argsort(np.array(times))
vortex_array = np.array(vortex_counts)[idx]
times_array = np.array(times)[idx]

sep = 12
logxi = np.linspace(np.min(np.log10(times_array)),np.max(np.log10(times_array)),20)
#logxi2 = np.linspace(np.min(np.log10(times_array[12:])),np.maxplt.(np.log10(times_array)),20)
#npan
plt.plot(np.log10(times_array), np.log10(vortex_array))

res, = plt.plot(logxi[:sep+1], -2.0*logxi[:sep+1]+5.3)
res.set_label('-2x+5.3')
res, = plt.plot(logxi[sep:], -1.8*logxi[sep:]+4.9)
res.set_label('-1.8x+5.3')
plt.legend()
plt.show()

#breakpoint()
    
