# Script used to convert epri5 loads script to loads and lines seperately.

import getpass
import os

if getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18\ieee_tn\ckt5\\"

os.chdir(WD)
loads_file = open('Loads_ckt5.dss')

loads = []
lines = []
i = 0
for line in loads_file:
    if len(line)>3:
        if line[0:8]=='New Line': # NB needs to pick up the linecode as well.
            lines=lines+[line]
        elif line[0:8]=='New Load':
            loads=loads+[line]
    i+=1

loads_file.close()

os.remove('Loads_ckt5_z.dss')
loads_file_out = open('Loads_ckt5_z.dss','x')
os.remove('Loads_lines_ckt5_z.dss')
lines_file_out = open('Loads_lines_ckt5_z.dss','x')


for load in loads:
    loads_file_out.write(load)
loads_file_out.close()

for line in lines:
    lines_file_out.write(line)

lines_file_out.close()