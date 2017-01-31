from algorithm.CSinput import get_files
import Constants as const
import json
import numpy as np

files = get_files("test_directory/", "final")
stat_files = get_files("nostrings/", "final")

tot_num = 1652.0

Q = 0
num = 0
sum = 0
sum_sq = 0

for file, stat_file in zip(files, stat_files):
    with open(file, 'r') as f:
        with open(stat_file, 'r') as g:
            for line1, line2 in zip(f, g):
                info = json.loads(line1[:-1])
                stats = json.loads(line2[:-1])
                
                measure = (5*const.deg)**2
                value = info['value']
                harmonic_value = np.sqrt(5/np.pi)/4 * (3*np.sin(info['theta'])**2 - 2)
                
                num += 1
                sum += stats['value']/tot_num
                sum_sq += stats['value']**2/tot_num
                
                Q += measure * value * harmonic_value

print Q, num
print np.sqrt(sum_sq - sum**2) * 5*const.deg
