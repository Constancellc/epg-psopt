import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

profiles = {}
ev_profiles = {}
with open('../../../Documents/pecan-street/1min-texas/aug-18.csv',
          'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        hh = row[1]+row[0][8:10]
        if hh not in profiles:
            profiles[hh] = [0.0]*1440
            ev_profiles[hh] = [0.0]*1440

        t = int(row[0][11:13])*60+int(row[0][14:16])

        profiles[hh][t] += float(row[3])-float(row[2])
        ev_profiles[hh][t] += float(row[2])

print(len(profiles))
for hh in profiles:
    plt.figure()
    plt.plot(profiles[hh])
    plt.plot(ev_profiles[hh])
    plt.show()
