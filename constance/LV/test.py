import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization_new import LVTestFeeder

network = LVTestFeeder('manc_models/1')
network.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv')
network.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                    'constance/ST1charges/')
b = network.get_feeder_load()
network.uncontrolled()
print(network.predict_losses)
p = network.get_feeder_load()
print(network.predict_losses)
network.load_flatten()
p2 = network.get_feeder_load()
print(network.predict_losses)
network.loss_minimise()
p3 = network.get_feeder_load()
print(network.predict_losses)

plt.figure()
plt.plot(b)
plt.plot(p)
plt.plot(p2)
plt.plot(p3)
plt.show()
