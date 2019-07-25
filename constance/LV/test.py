import csv
'''
days = {}
energy = {}
with open('../../../Documents/pecan-street/1min-texas/summer-18.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        hh = row[1]
        d = row[0][:10]
        p = float(row[3])-float(row[2])

        if hh not in days:
            days[hh] = []
            energy[hh] = 0

        if d not in days[hh]:
            days[hh].append(d)
        energy[hh] += p/60

av = 0
n = 0
for hh in energy:
    av += energy[hh]
    n += len(days[hh])

print(av/n)
'''

av = 0
with open('../../../Documents/netrev/TC2a/03-Dec-2013.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        n = len(row)-1
        for i in range(1,len(row)):
            av += float(row[i])/60
print(av/n)
