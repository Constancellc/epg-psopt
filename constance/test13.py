nPhases = 3

linePhases = {'601':[1,1,1],
              '602':[1,1,1],
              '603':[0,1,1],
              '604':[1,0,1],
              '605':[0,0,1]}

voltageSolutions = {650:[1.0,1.0,1.0], 
                    632:[1.021,1.042,1.0687],
                    633:[1.018,1.0401,1.0174],
                    634:[0.994,1.0218,0.9960],
                    645:[0,1.0329,1.0155],
                    646:[0,1.0311,1.0134],
                    671:[0.99,1.0529,0.9778],
                    680:[0.99,1.0529,0.9778],
                    684:[0.9881,0,0.9758],
                    611:[0,0,0.9738],
                    652:[0.9825,0,0],
                    692:[0.99,1.0529,0.9777],
                    675:[0.9835,1.0553,0.9758]}

nodeVoltageBases = {650:4160, # phases, Vbase
                    632:4160,
                    633:4160,
                    634:480,
                    645:4160,
                    646:4160,
                    671:4160,
                    680:4160,
                    684:4160,
                    611:4160,
                    652:4160,
                    692:4160,
                    675:4160}


nodes = {650:[1,1,1], # phases, Vbase
         632:[1,1,1],
         633:[1,1,1],
         634:[1,1,1],
         645:[0,1,1],
         646:[0,1,1],
         671:[1,1,1],
         680:[1,1,1],
         684:[1,0,1],
         611:[0,0,1],
         652:[1,0,0],
         692:[1,1,1],
         675:[1,1,1]}

slackbus = [650,122]

# Node A, Node B, Length (ft), Config, phases
lines = [[632,645,500,'603',[0,1,1]],
         [632,633,500,'602',[1,1,1]],
         [633,634,0,'XFM-1',[1,1,1]],
         [645,646,300,'603',[0,1,1]],
         [650,632,2000,'601',[1,1,1]],
         [684,652,800,'607',[1,0,0]],
         [632,671,2000,'601',[1,1,1]],
         [671,684,300,'604',[1,0,1]],
         [671,680,1000,'601',[1,1,1]],
         [671,692,0,'Switch',[1,1,1]],
         [684,611,300,'605',[0,0,1]],
         [692,675,500,'606',[1,1,1]],
         [671,675,500,'606',[1,1,1]]]


#nodes = {650:0,646:0,645:2,632:3,633:4,634:5,611:6,684:7,671:8,675:9,
#         652:10,680:11}

# Node, Phase 1 (kW), Phase 1 (kVar), Phase 2 (kW), Phase 2 (kVar),
# Phase 3 (kW), Phase 3 (kVar)
spotLoads = {634:[160000,110000,120000,90000,120000,90000],
             645:[0,0,170000,125000,0,0],
             646:[0,0,230000,132000,0,0],
             652:[128000,86000,0,0,0,0],
             671:[385000+9000,220000+5000,385000+33000,220000+19000,
                  385000+59000,220000+34000],
             675:[485000,190000-200000,68000,60000-200000,290000,
                  212000-200000],
             692:[0,0,0,0,170000,151000],
             611:[0,0,0,0,170000,80000-100000],
             632:[9000,5000,33000,19000,59000,34000]}

Z = {'601':[[complex(0.3465,1.0179),complex(0.1560,0.5017),
             complex(0.1580,0.4236)],
            [complex(0.3375,1.0478),complex(0.1535,0.3849)],
            [complex(0.3414,1.0348)]],
     '602':[[complex(0.7526,1.1814),complex(0.1580,0.4236),
             complex(0.1560,0.5017)],
            [complex(0.7475,1.1983),complex(0.1535,0.3849)],
            [complex(0.7436,1.2112)]],
     '603':[[complex(0.0000,0.0000),complex(0.0000,0.0000),
             complex(0.0000,0.0000)],
            [complex(1.3294,1.3471),complex(0.2066,0.4591)],
            [complex(1.3238,1.3569)]],
     '604':[[complex(1.3238,1.3569),complex(0.0000,0.0000),
             complex(0.2066,0.4591)],
            [complex(0.0000,0.0000),complex(0.0000,0.0000)],
            [complex(1.3294,1.3471)]],
     '605':[[complex(0.0000,0.0000),complex(0.0000,0.0000),
             complex(0.0000,0.0000)],
            [complex(0.0000,0.0000),complex(0.0000,0.0000)],
            [complex(1.3292,1.3475)]],
     '606':[[complex(0.7982,0.4463),complex(0.3192,0.0328),
             complex(0.2849,-0.0143)],
            [complex(0.7891,0.4041),complex(0.3192,0.0328)],
            [complex(0.7982,0.4463)]],
     '607':[[complex(1.3425,0.5124),complex(0.0000,0.0000),
             complex(0.0000,0.0000)],
            [complex(0.0000,0.0000),complex(0.0000,0.0000)],
            [complex(0.0000,0.0000)]],
     'Switch':[[complex(0.000,0.000),complex(0.0000,0.0000),
             complex(0.0000,0.0000)],
            [complex(0.0000,0.0000),complex(0.0000,0.0000)],
            [complex(0.0000,0.0000)]]}

transformers = {'XFM-1':[0.011,0.02]}
