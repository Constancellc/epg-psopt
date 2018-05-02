Folders:
1. 1ACPF-master is a direct copy of the Bolognani paper that we worked on for that linearization work.
2. method_comparison takes "1ACPF-master" works through and compares the different linearization methods.
3. source_impedance looks at how to 'include' the source impedance within OpenDSS (nominally the Ybus does not include the source impedance and so will give 'incorrect' results c.f. if it did include it within the Ybus)
4. 3phase_linearization is a (quite messy/old) folder that was used to try and linearise some of the ieee networks. Seems to work for the EU LV network (but doesnt for the IEEE 13 bus). Linearizes using the NREL method mostly.
5. fot_linearization attempts to take the FOT NREL method. Is currently not working, but it does appear (?) to allow for the FPL method.
MTD 01/05/18