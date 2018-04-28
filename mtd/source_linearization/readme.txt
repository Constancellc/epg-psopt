This folder is used to consider how we should account for source impedances within OpenDSS, how to calculate them and insert them into analysis so that we can consider all voltage drops. In other words, it allows for a more ideal voltage source approximation and admittance matrix, which is useful for network linearizations that DO account for this.

SCRIPTS:
1. run_test is used to show how source_impedance works, test.dss is used to play with exporting the sequence impedances of a lone voltage source.

2. compare_source is used to show how to implement the artificial source impedance files so these can be included in a lienarization properly (ie without cheating).