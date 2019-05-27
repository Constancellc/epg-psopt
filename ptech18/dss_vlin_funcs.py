import numpy as np
import os
from scipy import sparse
from cvxopt import matrix
import scipy.sparse.linalg as spla
from dss_python_funcs import *

def nrel_linearization(Ybus,Vh,V0,H):
    Yll = Ybus[3:,3:].tocsc()
    Yl0 = Ybus[3:,0:3].tocsc()
    H0 = sparse.csc_matrix(H[:,3:])
    
    Ylli = spla.inv(Yll)
    
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) ).tocsc() # NB: this looks slow
    Vh_diagi = spla.inv(Vh_diag)

    HVh_diag = sparse.dia_matrix( (H0.dot(Vh.conj()),0) ,shape=(H0.shape[0],H0.shape[0]) ).tocsc() # NB: this looks slow
    HVh_diagi = spla.inv(HVh_diag)
    
    My_0 = Ylli.dot(Vh_diagi)
    Md_0 = Ylli.dot(H0.T.dot(HVh_diagi))

    My = sparse.hstack((My_0,-1j*My_0)).toarray()
    Md = sparse.hstack((Md_0,-1j*Md_0)).toarray()
    
    a = -Ylli.dot(Yl0.dot(V0))
    
    return My,Md,a

def nrel_linearization_My(Ybus,Vh,V0):
    Yll = Ybus[3:,3:].tocsc()
    Yl0 = Ybus[3:,0:3].tocsc()
    a = spla.spsolve(Yll,Yl0.dot(-V0))
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    My_i = Vh_diag.dot(Yll)
    My_0 = spla.inv(My_i.tocsc())
    My = sparse.hstack((My_0,-1j*My_0)).toarray()
    return My,a

def nrel_linearization_Ky(My,Vh,sY):
    # an old version of this function; see nrelLinKy() below
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    Vhai_diag = sparse.dia_matrix( (np.ones(len(Vh))/abs(Vh),0),shape=(len(Vh),len(Vh)) )
    Ky = Vhai_diag.dot( Vh_diag.dot(My).real )
    b = abs(Vh) - Ky.dot(-1e3*s_2_x(sY[3:]))
    return Ky, b

def nrel_linearization_K(My,Md,Vh,sY,sD):
    # an old version of this function; see nrelLinKy() below
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    Vhai_diag = sparse.dia_matrix( (np.ones(len(Vh))/abs(Vh),0),shape=(len(Vh),len(Vh)) )
    Ky = Vhai_diag.dot( Vh_diag.dot(My).real )
    Kd = Vhai_diag.dot( Vh_diag.dot(Md).real )
    b = abs(Vh) - Ky.dot(-1e3*s_2_x(sY[3:]))- Kd.dot(-1e3*s_2_x(sD))
    return Ky, Kd, b

def nrelLinKy(My,Vh,xY):
    # based on nrel_linearization_Ky
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    Vhai_diag = sparse.dia_matrix( (np.ones(len(Vh))/abs(Vh),0),shape=(len(Vh),len(Vh)) )
    Ky = Vhai_diag.dot( Vh_diag.dot(My).real )
    b = abs(Vh) - Ky.dot(xY)
    return Ky, b

def nrelLinK(My,Md,Vh,xY,xD):
    # based on nrel_linearization_K
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    Vhai_diag = sparse.dia_matrix( (np.ones(len(Vh))/abs(Vh),0),shape=(len(Vh),len(Vh)) )
    Ky = Vhai_diag.dot( Vh_diag.dot(My).real )
    Kd = Vhai_diag.dot( Vh_diag.dot(Md).real )
    b = abs(Vh) - Ky.dot(xY) - Kd.dot(xD)
    return Ky, Kd, b

def lineariseMfull(My,Md,Mt,f0,xY,xD,xT):
    # based on nrelLinK
    f0_diag = sparse.dia_matrix( (f0.conj(),0),shape=(len(f0),len(f0)) )
    f0ai_diag = sparse.dia_matrix( (np.ones(len(f0))/abs(f0),0),shape=(len(f0),len(f0)) )
    Ky = f0ai_diag.dot( f0_diag.dot(My).real )
    Kd = f0ai_diag.dot( f0_diag.dot(Md).real )
    Kt = f0ai_diag.dot( f0_diag.dot(Mt).real )
    b = abs(f0) - Ky.dot(xY) - Kd.dot(xD) - Kt.dot(xT)
    return Ky, Kd, Kt, b

def fixed_point_itr(w,Ylli,V,sY,sD,H):
    iTtot_c = sY/V + H.T.dot(sD/(H.dot(V)))
    V = w + Ylli.dot(iTtot_c.conj())
    return V

def fixed_point_solve(Ybus,YNodeV,sY,sD,H): # seems to give comparable results to opendss.
    v0 = YNodeV[0:3]
    Yl0 = Ybus[3:,0:3]
    Yll = Ybus[3:,3:]
    Ylli = spla.inv(Yll)
    w = -Ylli.dot(Yl0.dot(v0))
    dV = 1
    eps = 1e-10
    V0 = w
    while (np.linalg.norm(dV)/np.linalg.norm(w))>eps:
        V1 = fixed_point_itr(w,Ylli,V0,sY,sD,H[:,3:])
        dV = V1 - V0
        V0 = V1
    return V0
    
    
def cvrLinearization(Ybus,Vh,V0,H,pCvr,qCvr,kvYbase,kvDbase):
    # based on nrel_linearization.
    # Y-bit:
    Yll = Ybus[3:,3:].tocsc()
    Yl0 = Ybus[3:,0:3].tocsc()
    H0 = sparse.csc_matrix(H[:,3:])
    
    Ylli = spla.inv(Yll)
    
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) ).tocsc() # NB: this looks slow
    Vh_diagi = spla.inv(Vh_diag)

    HVh_diag = sparse.dia_matrix( (H0.dot(Vh.conj()),0) ,shape=(H0.shape[0],H0.shape[0]) ).tocsc() # NB: this looks slow
    try:
        HVh_diagi = spla.inv(HVh_diag)
    except:
        HVh_diagi = H0
    
    pYcvr_diag = sparse.dia_matrix( ( abs(Vh/kvYbase)**pCvr,0 ),shape=(len(Vh),len(Vh)) ).tocsc()
    qYcvr_diag = sparse.dia_matrix( ( abs(Vh/kvYbase)**qCvr,0 ),shape=(len(Vh),len(Vh)) ).tocsc() 
    pDcvr_diag = sparse.dia_matrix( ( abs(H0.dot(Vh)/kvDbase)**pCvr,0 ),shape=(H0.shape[0],H0.shape[0]) ).tocsc() 
    qDcvr_diag = sparse.dia_matrix( ( abs(H0.dot(Vh)/kvDbase)**qCvr,0 ),shape=(H0.shape[0],H0.shape[0]) ).tocsc() 
    
    My_0 = Ylli.dot(Vh_diagi)
    Md_0 = Ylli.dot(H0.T.dot(HVh_diagi))

    My = sparse.hstack((My_0.dot(pYcvr_diag),-1j*My_0.dot(qYcvr_diag))).toarray()
    try:
        Md = sparse.hstack((Md_0.dot(pDcvr_diag),-1j*Md_0.dot(qDcvr_diag))).toarray()
    except:
        Md = H0.T
    
    a = -Ylli.dot(Yl0.dot(V0))
    
    # D-bit:
    dMy = H0.dot(My)
    dMd = H0.dot(Md)
    da = H0.dot(a)
    
    return My,Md,a,dMy,dMd,da
    
def pdTest(A):
    # first find the symmetric part of A, then try the cholesky decomposition.
    S = 0.5*(A + A.T)
    try:
        np.linalg.cholesky(S)
        pd = True
    except:
        pd = False
    return pd
    
    