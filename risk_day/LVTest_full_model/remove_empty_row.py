import csv

file = 'riskDayLoadIn.csv'

data = []

with open(file,'rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row != []:
            data.append(row)

with open(file,'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in data:
        writer.writerow(row)
