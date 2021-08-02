# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 08:59:15 2021

@author: mnguy
"""


import cvxpy as cp
import numpy as np
import scipy
from scipy import linalg, optimize
import cvxpy.atoms.elementwise.log as cplog
import cvxpy.atoms.elementwise.sqrt as cpsqrt
import cvxpy.atoms.affine.sum as cpsum
import cvxpy.atoms.affine.binary_operators as cpoperate
import cvxpy.atoms.elementwise.exp as cpexp
import cvxopt

###------------------------------------------------------------------------------
def create_position(N_APs, N_UEs, sq_length):
    
    # N_APs: number of APs
    M = N_APs;  
    step_len = sq_length/(np.sqrt(M)-1)
    m1 = int(np.floor(np.sqrt(M)))
    m2 = int(M-m1**2)
    
    posAP = np.zeros((M,2))
    t = 0;
    for m in range(m1):
        for n in range(m1):
            posAP[t,0] = step_len*m + step_len/2
            posAP[t,1] = step_len*n + step_len/2
            t = t+1;
    if m2>0:
        for m in range(m2):            
            posAP[t,0] = step_len*m
            posAP[t,1] = step_len*m1
            t = t+1;
    # uniform random UE length
    
    m = 0
    K = N_UEs[m]
    pUE = step_len*(np.random.rand(K,2)*0.9+0.1)  
    pUE[:,0] = pUE[:,0] + posAP[m,0] - step_len/2
    pUE[:,1] = pUE[:,1] + posAP[m,1] - step_len/2
    posUE = np.copy(pUE)
    for m in range(1,posAP.shape[0]):
        K = N_UEs[m]
        pUE = step_len*(np.random.rand(K,2)*0.9+0.1)
        
        pUE[:,0] = pUE[:,0] + posAP[m,0] - step_len/2
        pUE[:,1] = pUE[:,1] + posAP[m,1] - step_len/2
        posUE = np.vstack((posUE,pUE))
        
            
    return posAP, posUE

###----------------------------------------------------------------------------
def create_large_scale(posAP, posUE, ptr, tauH):
    # posAP: a vector location of APs: B x 2
    # posUE: a vector location of UE: K x 2
    # ptr: training power: K x 1
    # tauH: size B x 1
    # tc, c, gammaH, betaH : size B x K
    
    
    normalized_p = 1e14
    B = posAP.shape[0]
    K = posUE.shape[0]
    
    Duser = np.zeros((B,K))
    

    for b in range(B):    
        for k in range(K):
            Duser[b,k] = np.sqrt(np.sum((posAP[b] - posUE[k])**2))
    H1 = 61.4+34*np.log10(Duser)
    betaH = 10**(-H1/10)*normalized_p  
    
    
    gammaH = np.copy(betaH)
    tc = np.copy(betaH)
    c = np.copy(betaH)
    
    for b in range(B):
        for k in range(K):
            tc[b,k] = betaH[b,k]/(tauH[b]*ptr[k]*betaH[b,k]+1)
            c[b,k] = tc[b,k]*np.sqrt(tauH[b]*ptr[k]);
            gammaH[b,k] = c[b,k]*np.sqrt(tauH[b]*ptr[k])*betaH[b,k];
    
    return betaH, tc ,c, gammaH


###----------------------------------------------------------------------------
def get_tauH(N_UEs, x_place, B):
    tauH = np.copy(N_UEs)
    tau0 = np.dot((1-x_place),N_UEs)
    for b in range(B):
        if x_place[b] < 1:
            tauH[b] = tau0
            
    return tauH


###----------------------------------------------------------------------------
def get_quantized_coef(h_bit, y_bit, B):
    # y_bit; a vector of B x 1
    # h_bit: a list, element b of the list is a vector of B x K_b
    a_L = np.array([0.6366, 0.88115, 0.96256, 0.98845, 0.99651,0.99896, 0.99969, 0.99991, 0.999975, 0.9999931])
    s_L = np.array([0.2313, 0.10472, 0.036037, 0.011409, 0.003482, 0.0010289, 0.0003042, 0.0000876, 2.492*1e-5, 6.997*1e-6])
    
    ah = a_L[h_bit-1]
    sh = s_L[h_bit-1]
    
    ay = np.array([])
    sy = np.array([])
    
    for b in range(B):
        ay = np.append(ay, a_L[y_bit[b]-1])
        sy = np.append(sy, s_L[y_bit[b]-1])
        
    return ay, sy, ah, sh



def compute_rate_coef1(x_place, betaH, gammaH, p, p0, M, CumUE, N_UEs, IndUE2):
    B = betaH.shape[0]
    K = betaH.shape[1]
    
    S = []
    for b in range(B):
        S1 = 1
        for k in range(CumUE[b],CumUE[b+1]):
            S1 += p[k]*(betaH[b,k] - gammaH[b,k])
        S.append(S1)
            
    
    DS = []
    IUE = []
    for k in range(K):
        b = int(IndUE2[k])
        DS.append(gammaH[b,k]*(M[b] - N_UEs[b])*p[k])
        IUE.append(S[b])
    
        
        
    S0 = []
    for b in range(B):
        S10 = 1
        for k in range(CumUE[b],CumUE[b+1]):
            S10 += p0[k]*(betaH[b,k] - gammaH[b,k])
        S0.append(S10)
    
    inlogR1 = []
    for k in range(K):
        b = int(IndUE2[k])
        inlogR1.append(S[b] + DS[k])
    
    appr_R2 = []
    for b in range(B):        
        appr_R2.append(np.log2(S0[b]) + (S[b] - S0[b])/S0[b])
        
    return DS, IUE, inlogR1, appr_R2


def get_rate1(DS, IUE, inlogR1, appr_R2, K, IndUE2, flag_R):
    
    R1 = []
    for k in range(K):
        b = int(IndUE2[k])
        R1.append(cplog.log(inlogR1[k])/cplog.log(2) - appr_R2[b])
    
    R1t = 1 
    if flag_R: # true value of R1
        R1t = []
        for k in range(K):
            R1t.append(np.log2(DS[k]/IUE[k]))
        
        

    return R1, R1t
        



    
def compute_rate_coef0(x_place, betaH, gammaH, p, p0, CumUE, N_UEs, IndUE2, h_bit, y_bit, M ):
    ay, sy, ah, sh = get_quantized_coef(h_bit, y_bit, B)
    
    setB0 = np.array([])
    setK0 = np.array([])
    for b in range(B):
        if x_place[b] == 0:
            setB0 = np.append(setB0,b)
            setK0 = np.append(setK0,np.array(range(CumUE[b],CumUE[b+1])))
    
    
    M0 = np.dot(M, 1-x_place)
    K0 = setK0.shape[0]
    S = [] # list size B0 x 1
    S0 = []
    for b1 in range(setB0.shape[0]):
        b = int(setB0[b1])
        S1 = ay[b]**2 + sy[b]**2
        S10 = ay[b]**2 + sy[b]**2
        for j1 in range(K0):            
            j = int(setK0[j1])
            S1 += p[j]*ay[b]**2*(betaH[b,j] - gammaH[b,j])
            S1 += p[j]*sy[b]**2*betaH[b,j]
            S10 += p0[j]*ay[b]**2*(betaH[b,j] - gammaH[b,j])
            S10 += p0[j]*sy[b]**2*betaH[b,j]
        S.append(M0/(M0-K0+1)*S1)
        S0.append(M0/(M0-K0+1)*S10)
    
    
    R = []
    for k in range(K):
        DS1 = 0
        for b1 in range(setB0.shape[0]):
            b = int(setB0[b1])
            DS1 += p[k]*(M[b]-1)*gammaH[b,k]/S[b1]
            
        R.append(DS1)
    
    return S, S0, R


        
def solve_case0(x_place, betaH, gammaH, p0, CumUE,  N_UEs, IndUE2, h_bit, y_bit, M, Rkmin, W0, pmax):
    
    
    
    B = betaH.shape[0]
    
    Rkmin0 = np.array([]) 
    ay, sy, ah, sh = get_quantized_coef(h_bit, y_bit, B)
    
    setB0 = np.array([])
    setK0 = np.array([])
    tp0 = np.array([])
    for b in range(B):
        if x_place[b] == 0:
            setB0 = np.append(setB0,b)
            setK0 = np.append(setK0,np.array(range(CumUE[b],CumUE[b+1])))
            Rkmin0 = np.append(Rkmin0, Rkmin[CumUE[b]:CumUE[b+1]])
            tp0 = np.append(tp0, p0[CumUE[b]:CumUE[b+1]])
    
    M0 = np.dot(M, 1-x_place)
    B0 = setB0.shape[0]
    K0 = setK0.shape[0]
    
    
    
    
    z0 = np.zeros((B0,K0))
    for b1 in range(setB0.shape[0]):
        b = int(setB0[b1])
        for k1 in range(K0):          
            
            zz1 = (1+sy[b]**2/ay[b]**2) 
            for j1 in range(K0):
                j = int(setK0[j1])
                zz1 += (sy[b]**2/ay[b]**2*betaH[b,j] + betaH[b,j]-gammaH[b,j])*tp0[j1]
            z0[b1,k1] = tp0[k1]/zz1
    
    f1 = (M0-K0+1)/M0
                
    for tt in range(5):
    
        phi0 = np.zeros((B0,K0,K0))
        for b1 in range(B0):
            for k1 in range(K0):
                for j1 in range(K0):
                    phi0[b1,k1,j1] = tp0[j1]/z0[b1,k1]
        
        
        
        
        
    
    
        p = cp.Variable(K0, nonneg=True)
        z = cp.Variable((B0,K0), nonneg=True)
        
        eps = cp.Variable()
        #p = np.copy(tp0)
        #z = np.copy(z0)
        
        
        
        R = np.array([])
        for j1 in range(K0):            
            j = int(setK0[j1])
            
            A1 = 0
            for b1 in range(setB0.shape[0]):
                b = int(setB0[b1])
                A1 += (M[b]-1)*gammaH[b,j]*z[b1,j1]
            
            R1 = W0*cplog.log(1+f1*A1)/np.log(2)
            R = np.append(R,R1)
            
        
        constraints = []
            
                
        for b1 in range(setB0.shape[0]):
            b = int(setB0[b1])
            for k1 in range(K0):          
                
                SS = z[b1,k1]*(1+sy[b]**2/ay[b]**2) 
                for j1 in range(K0):
                    j = int(setK0[j1])
                    SS += (sy[b]**2/ay[b]**2*betaH[b,j] + betaH[b,j]-gammaH[b,j])*(phi0[b1,k1,j1]*z[b1,k1]**2/2 + p[j1]**2/(2*phi0[b1,k1,j1]))
                    
                    constraints.append(SS <= p[k1])
        
        obj1 = 0
        for k in range(K0):
            constraints.append(p[k] <= pmax)
            obj1 += R[k]
        
        tflag = True   
        if tflag:
            for k in range(K0):
                constraints.append(R[k] >= Rkmin0[k] - 0*eps)
            
            
        obj = cp.Maximize(obj1 - 0*eps)   
        prob = cp.Problem(obj, constraints)
        #prob.solve(solver=cp.MOSEK)
        prob.solve(solver=cp.MOSEK)        
        
        for k in range(K0):
            tp0[k] = p[k].value
            
            
        for b1 in range(setB0.shape[0]):
            for k1 in range(K0):
                z0[b1,k1] = z[b1,k1].value
                
                
        print( 'obj {}'.format(obj1.value - 100*eps.value))
        
        
    
        
    return p.value
            
            
            
            
            
    

    
            
            
            
            
    
        
    
        
        




###----------------------------------------------------------------------------
N_UEs = np.array([5,6,7,4])
N_APs = N_UEs.shape[0]
B = N_APs
K = np.sum(N_UEs)
M = 16*np.ones(B)
sq_length = 1000
ptr = 1e4*np.ones(K)


IndUE = np.zeros((B,K))
CumUE = np.append(0,np.cumsum(N_UEs))

for b in range(B):
    IndUE[b,CumUE[b]:CumUE[b+1]]=1

IndUE2 = np.zeros(K)
for b in range(B):
    IndUE2[CumUE[b]:CumUE[b+1]]=b
    
    
posAP, posUE = create_position(N_APs, N_UEs, sq_length)


x_place = np.random.randint(2,size = B)
tauH = get_tauH(N_UEs, x_place, B);    
betaH, tc ,c, gammaH = create_large_scale(posAP, posUE, ptr, tauH)

p0 = 1e2*np.ones(K) + 1e1*np.random.rand(K)
p = 2*1e2*np.ones(K)

p = np.copy(p0)

h_bit = 10
y_bit = np.random.randint(3,5, size = B)

ay, sy, ah, sh = get_quantized_coef(h_bit, y_bit, B) # ay,sy: size B x 1; ah,sh: size 1 x 1

DS1, IUE1, inlogR11, appr_R21 = compute_rate_coef1(x_place, betaH, gammaH, p, p0, M, CumUE, N_UEs, IndUE2)

R1, R1t = get_rate1(DS1, IUE1, inlogR11, appr_R21, K, IndUE2, True)

S, S0, R = compute_rate_coef0(x_place, betaH, gammaH, p, p0, CumUE, N_UEs, IndUE2, h_bit, y_bit, M )

Rkmin = 4*np.ones(K)
W0=1

pmax = 1.5*1e2

P = solve_case0(x_place, betaH, gammaH, p0, CumUE,  N_UEs, IndUE2, h_bit, y_bit, M, Rkmin, W0, pmax)
    
for b in range(B):
    if x_place[b] == 1:
        P1 = P[:CumUE[b]]
        P2 = P[CumUE[b]:]
        
        P3 = np.append(P1, p0[CumUE[b]:CumUE[b+1]])
        P3 = np.append(P3,P2)
        
        P = np.copy(P3)

    
S, S0, R = compute_rate_coef0(x_place, betaH, gammaH, P, p0, CumUE, N_UEs, IndUE2, h_bit, y_bit, M )

