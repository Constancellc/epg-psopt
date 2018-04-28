3phase_linearization is a folder that is used to create and test various linearization codes.

SCRIPTS:
XX. 4bus_powerflows.m is used to fine all of the sY sD iY iD power flows within opendss that are requried for the linearisation. Is missing a script iD_iY for what it is supposed to be doing. (?)

1.1 reproduce_ieee13.m reproduces the ieee13 bus results with opendss (with the finding that the frequency must be chosen carefully!)

1.2 reproduce_implicit15.m attempts to reproduce the results from the bolognani paper, and demonstrates that they appear to be cheating slightly in that work.


0. 4bus_linearization.m is a script that linearizes all of the different 4 bus cases. This seems to not work quite as well as the other case studies, perhaps due to capacitance in the lines? 

0. create_lvtestcase.m demonstrates the linearization working for the eu lv test case for both losses and voltages.

0. reproduce_nrel.m is a script that looks to reproduce the results of the paper "load-flow in multiphase distrbution networks: existence, uniqueness, and linear models", available at https://arxiv.org/abs/1702.03310 . does so for the delta connected 37 bus feeder.

0. run_linearization is (a rather old) script that gets the 13 bus linearization up and running. It then checks the linearization at a particular bus (rather than doing a continuation analysis).

