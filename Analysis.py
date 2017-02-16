import json

import matplotlib.pyplot as plt
import numpy as np

import Constants as const
from algorithm.DirectoryEmbedded import get_files

files = get_files("test_directory/", "final")
stat_files = get_files("nostrings/", "final")

tot_num = 1652.0

Q = 0
num = 0
sum = 0
sum_sq = 0
vals_str =  []
vals_nostr = []

for file, stat_file in zip(files, stat_files):
    with open(file, 'r') as f:
        with open(stat_file, 'r') as g:
            for line1, line2 in zip(f, g):
                info = json.loads(line1[:-1])
                stats = json.loads(line2[:-1])
                
                measure = (5*const.deg)**2
                value = info['value']
                theta = info['theta']
                harmonic_value = np.sqrt(5/np.pi)/4 * (3*np.sin(info['theta'])**2 - 2)
                
                if abs(theta - np.pi/2) < np.pi/6:
                    vals_str.append(value)
                elif abs(theta - np.pi/2) > np.pi/3:
                    vals_nostr.append(value)
                
                num += 1
                sum += stats['value']/tot_num
                sum_sq += stats['value']**2/tot_num
                
                Q += measure * value * harmonic_value

print Q, num
print np.sqrt(sum_sq - sum**2) * 5*const.deg

plt.hist(np.array(vals_str), 20, normed=1, facecolor='green', alpha=0.75)
plt.hist(np.array(vals_nostr), 20, normed=1, facecolor='blue', alpha=0.75)

plt.title("Distribution of Scalar output with strings (Green) and without (Blue)")
plt.ylabel("Normed Probability")
plt.xlabel("Scalar Ouptut from Learning Algorithm")
plt.show()
