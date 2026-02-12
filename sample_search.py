# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.linalg import solve
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from functools import reduce
from numpy.linalg import inv
import plotly.graph_objects as go

γe = 1.76086e8
σ = [
    [0.5 * np.array([[0, 1], [1, 0]]),
    0.5 * np.array([[0, -1j], [1j, 0]]),
    0.5 * np.array([[1, 0], [0, -1]])],
    [(1/np.sqrt(2)) * np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]]),
     (1/np.sqrt(2)) * np.array([[0, -1j, 0],[1j, 0, -1j],[0, 1j, 0]]),
     np.array([[1, 0, 0],[0, 0, 0],[0, 0, -1]])]
]

def tensor_product(*args):
    return reduce(np.kron, args)

def Id(d):
    return np.identity(d)
def product(*args):
  return reduce(np.multiply,*args)

def HyperfineHamiltonian(n, spin, hfc):
    if n == 0:
      M = 1
      return 1
    else:
      M = product(spin)
      II=[]
      for j in range(n):
        III=[]
        for q in range(3):
          left_ops = Id(product(spin[:j])) if j>0 else 1
          right_ops = Id(product(spin[j+1:])) if j+1<n else 1
          III.append(tensor_product(left_ops,σ[spin[j]-2][q],right_ops))
        II.append(III)
      H=np.zeros((2*M,2*M),dtype=complex)
      for j in range(n):
        for p in range(3):
          for q in range(3):
            H += hfc[j][p][q]*tensor_product(σ[0][p],II[j][q])

    return H

def ZeemanHamiltonian(ω0, θ0, φ0, n,spin):
    if n == 0:
      M = 1
      S = σ[0]
    else:
      M = product(spin)
      S = [tensor_product(σ[0][i],Id(M)) for i in range(3)]
    term_x = S[0] * np.sin(θ0) * np.cos(φ0)
    term_y = S[1] * np.sin(θ0) * np.sin(φ0)
    term_z = S[2] * np.cos(θ0)
    return ω0 * (term_x + term_y + term_z)

def TY(nA, spinA, hfcA, nB, spinB, hfcB, ω0, θ0, φ0,ks,kt,rA,rB):

    HA = HyperfineHamiltonian(nA, spinA, hfcA) + ZeemanHamiltonian(ω0, θ0, φ0, nA, spinA)
    if nA == 0:
      MA = 1
      SA = σ[0]
    else:
      MA = product(spinA)
      SA = [tensor_product(σ[0][i],Id(MA)) for i in range(3)]

    HB = HyperfineHamiltonian(nB, spinB, hfcB) + ZeemanHamiltonian(ω0, θ0, φ0, nB, spinB)
    MB = HB.shape[0]
    if nB == 0:
      MB = 1
      SB = σ[0]
    else:
      MB = product(spinB)
      SB = [tensor_product(σ[0][i],Id(MB)) for i in range(3)]

    M = MA * MB
    H = np.kron(HA, Id(2*MB)) + np.kron(Id(2*MA), HB)

    PS = 0.25 * np.identity(4*M) - tensor_product(SA[0],SB[0]) - tensor_product(SA[1],SB[1]) - tensor_product(SA[2],SB[2])

    PT = np.identity(4*M) - PS

    IS = (1 / (3 * M)) * PT
    ISV = IS.flatten()[:, np.newaxis]


    KS = (ks / 2) * (np.kron(PS, Id(4*M)) + np.kron(Id(4*M), PS.T))
    KS = sp.csr_matrix(KS)

    KT = (kt / 2) * (np.kron(PT, Id(4*M)) + np.kron(Id(4*M), PT.T))
    KT = sp.csr_matrix(KT)

    RR = rA * (0.75 * np.kron(Id(4*M), Id(4*M)) - np.kron(np.kron(SA[0],Id(2*MB)),np.kron(SA[0],Id(2*MB)).T) -np.kron(np.kron(SA[1],Id(2*MB)),np.kron(SA[1],Id(2*MB)).T) - np.kron(np.kron(SA[2],Id(2*MB)),np.kron(SA[2],Id(2*MB)).T))+ rB * (0.75 * np.kron(Id(4*M), Id(4*M)) - np.kron(np.kron(Id(2*MA),SB[0]),np.kron(Id(2*MA),SB[0]).T) -np.kron(np.kron(Id(2*MA),SB[1]),np.kron(Id(2*MA),SB[1]).T) - np.kron(np.kron(Id(2*MA),SB[2]),np.kron(Id(2*MA),SB[2]).T))
    RR = sp.csr_matrix(RR)

    HH = np.kron(H, Id(4*M)) - np.kron(Id(4*M), H.T)
    HH = sp.csr_matrix(HH)
    LL = 1j * HH + KS + KT + RR
    rhov = spsolve(LL, ISV)
    rho = rhov.reshape((4*M, 4*M))
    yield_triplet = kt * np.real(np.trace(PT @ rho))
    return yield_triplet

def TY_pre(nA, spinA, hfcA, nB, spinB, hfcB, ω0, θ0, φ0,ks,kt,rA,rB):
  GMF = TY(nA, spinA, hfcA, nB, spinB, hfcB, 0.045 * γe, θ0, φ0,ks,kt,rA,rB)
  return (100*(TY(nA, spinA, hfcA, nB, spinB, hfcB, ω0, θ0, φ0,ks,kt,rA,rB)-GMF))/GMF

def f(B,a1,a2,b1,ks,kt,rA,rB,A):
  nA = 2
  nB = 1
  spinA = [2,3]
  spinB = [2]
  hfcA = [a1*γe*np.array([
    [1,  0.0,  0.0],
    [ 0.0, 1,  0.0],
    [ 0.0,     0.0,    1]]),a2*γe*np.array([
    [1,  0.0,  0.0],
    [ 0.0, 1,  0.0],
    [ 0.0,     0.0,    1]])]

  hfcB = [b1*γe*np.array([
    [1.0, 0.0,    0.0],
    [0.0,    1.0, 0.0],
    [0.0,    0.0,    1.0]])]
  return A*TY_pre(nA, spinA, hfcA, nB, spinB, hfcB, B * γe, 0, 0,ks,kt,rA,rB)

B_vals = np.array([0.0, 0.2, 0.5, 0.7, 0.9])
y_exp = np.array([68, -34, 52, 71, 130])
y_err0 = np.array([11, 4, 11, 9, 19])
y_err1 = np.array([258, 41, 141, 181, 515])
y_err2 = np.array([-74, -91, -39, -23, 11])
y_err3 = np.array([101, -13, 102, 104, 178])
y_err4 = np.array([24, -51, -4, 28, 57])
y_err00 = np.vstack((y_exp-y_err4, y_err3-y_exp))
y_err = np.vstack((y_exp-y_err2, y_err1-y_exp))

N_samples = 2400
a1_vals = np.random.choice([1], N_samples)
a2_vals = np.random.choice([1,0.1], N_samples)
b1_vals = np.random.choice([0,0.1,1,10], N_samples)
ks_vals = 10 ** np.random.uniform(3, 8, N_samples)
kt_vals = 10 ** np.random.uniform(3, 8, N_samples)
rA_vals = np.random.choice([1e6], N_samples)
rB_vals = np.random.choice([1e6], N_samples)
A_vals = np.random.choice([1, 10, 100], N_samples)

valid_params = []

for i, (a1,a2,b1,ks,kt,rA,rB,A) in enumerate(zip(a1_vals,a2_vals,b1_vals,ks_vals,kt_vals,rA_vals,rB_vals,A_vals),start=1):
    valid = True
    for B, y, err1, err2 in zip(B_vals, y_exp, y_err1, y_err2):
        y_theory = f(B,a1,a2,b1,ks,kt,rA,rB,A)
        if not (err2 <= y_theory <= err1):
            valid = False
            break
    if valid:
        valid_params.append((a1,a2,b1,ks,kt,rA,rB,A))


    if i % 100 == 0:
        print(f"Checked {i} trials... Found {len(valid_params)} valid so far.")

print(f"Found {len(valid_params)} valid parameter sets")
if len(valid_params) > 0:
    print(valid_params)

valid_params = np.array(valid_params)

for i, p in enumerate(valid_params):
    print(f"#{i}:a2={p[1]}, b={p[2]}, ks={p[3]:.2e}, kt={p[4]:.2e}, A={p[7]}")

for i, params in enumerate(valid_params):
        a1,a2,b1,ks,kt,rA,rB,A = params
        y_theory = [f(B,a1,a2,b1,ks,kt,rA,rB,A) for B in np.array([0.0,0.2, 0.5, 0.7, 0.9])]
        print(f"#{i}: Distance = {np.sqrt(np.sum((((y_theory-y_exp)**2)))):.2f}")

