nPhases = 3

slackbus = [1,12470]

nodes = {1:[1,1,1],2:[1,1,1],3:[1,1,1],4:[1,1,1]}

lines = [[1,2,2000,'d',[1,1,1]], # node a, node b, length (ft), config
         [2,3,0,'step-down',[1,1,1]],
         [3,4,2500,'d',[1,1,1]]]

spotLoads = {4:[1800000,0,1800000,0,1800000,0]}

Z = {'d':[[complex(0.4576,1.078),complex(0.1559,0.5017),
           complex(0.1535,0.3849)],
          [complex(0.1559,0.5017),complex(0.4666,1.0482)],
          [complex(0.1535,0.3849)]]}

transformers = {'step-down':[0.2592,1.5552,12.47/4.16]} # R, X, a
