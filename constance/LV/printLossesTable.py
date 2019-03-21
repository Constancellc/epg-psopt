import csv


stem = '../../../Documents/simulation_results/LV/manc-models/'
fds = ['041','213','162','1','3','2','193','074','024']

hhs = {'1':55,'2':175,'3':94,'4':24,'024':115,'041':24,'074':186,'162':73,
       '193':65,'213':67}

for fdr in fds:
    b = []
    u = []
    f = []
    m = []
    p = []
    
    with open(stem+fdr+'-losses.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            b.append(1000*float(row[0]))
            u.append(1000*float(row[1]))
            f.append(1000*float(row[2]))
            m.append(1000*float(row[3]))
            p.append(1000*float(row[4]))

    print(str(int(sum(b)/(len(b)*hhs[fdr])))+' & '+
          str(int(sum(u)/(len(b)*hhs[fdr])))+' & '+
          str(int(sum(f)/(len(b)*hhs[fdr])))+' & '+
          str(int(sum(m)/(len(b)*hhs[fdr])))+' & '+
          str(int(sum(p)/(len(b)*hhs[fdr]))))
    print('')
