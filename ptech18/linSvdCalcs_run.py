import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import gamma
import dss_stats_funcs as dsf
from linSvdCalcs import exampleClass, linModel

WD = os.path.dirname(sys.argv[0])

fdr_i = 22
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
feeder = fdrs[fdr_i]

netModel = 0
netModel = 1
netModel = 2

qwe = linModel(feeder,WD)
qwe.loadNetModel(netModel)

qwe.plotFlatVoltage()
