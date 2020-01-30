import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization import LVTestFeeder
import time

fdr = '1' # name of folder within manc_models - this one ie EULV
tr = 1 # time resolution in minutes

network = LVTestFeeder('manc_models/'+fdr,t_res=tr)

# below is a hack as I don't have the actual phases for the EULV to hand
phases = []
for i in range(18):
    phases += ['A','B','C']
phases += ['A']

network.set_households_synthetic(4)
network.set_evs_synthetic(5)
network.load_flatten()
p1 = network.get_feeder_load()

network.loss_minimise()
p2 = network.get_feeder_load()

network.balance_phase2(phases)
p3 = network.get_feeder_load()

plt.figure()
plt.plot(p1)
plt.plot(p2)
plt.plot(p3)
plt.show()
