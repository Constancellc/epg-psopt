from pypower.api import case14, ppoption, runpf, printpf

ppc = case14()
ppopt = ppoption(PF_ALG=2)
r = runpf(ppc,ppopt)
printpf(r)
