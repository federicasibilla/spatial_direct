"""
__initCC__.py: 

"""

import numpy as np

from R_dynamics import *
from N_dynamics import *
from visualization  import *
from simulate import *

# initialize R0
n_r = 3
n_s = 3
n   = 40 

R0  = np.zeros((n, n, n_r))
R0[:,:,0]=10
R0[:,:,1]=0
R0[:,:,2]=0

g   = np.array([0.5,0.5,0.8]) 
m   = np.array([0.,0.,0.])

# initialize species grid: random
N0 = np.zeros((n,n,n_s))
for i in range(n):
    for j in range(n):
        idx = np.random.randint(3)
        N0[i,j,idx]=1

# define parameters
param = {
    # model parameters
    'R0' : R0.copy(),                                  # initial conc. nxnxn_r [monod constants]
    'N0' : N0.copy(),                                  # initial species grid nxnxn_s
    'w'  : np.ones((n_r))*20,                          # energy conversion     [energy/mass]
    'l'  : np.ones((n_r))-0.2,                         # leakage               [adim]
    'tau': np.array([1,1000000,1000000]),              # reinsertion rate inv. [time] 
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    
    # discretization parameters
    'n'  : n,                                          # grid points in each dim
    'L'  : 40,                                         # grid true size        [length]
    'D'  : 1e3                                         # diffusion constant    [area/time] 
}

# make matrices
up_mat   = np.array([[1,0.,1],[1.,0.,1],[0.,1,0.]])
met_mat  = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
sign_mat = np.array([[1.,1.,1],[1.,1.,1],[1,1,1]])
mat_ess  = np.array([[0.,0.,1],[0.,0.,1],[0.,1,0.]])
spec_met = np.array([[0.,0.,0.],[0.,1,0.],[0.,0.,1.]])
print(up_mat)
print(met_mat)
print(sign_mat)
print(mat_ess)
print(spec_met)

mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'sign'    : sign_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met
}

# visualize matrices
vispreferences(mat)
makenet(met_mat)

#-------------------------------------------------------------------------------------------------------------
# SIMULATION

# run 1000 steps 
steps,R_fin,N_fin = simulate(1000,param,mat)

# plot final R and N grids
R_ongrid(R_fin)
N_ongrid(N_fin)
R_ongrid_3D((R_fin))

# produce animation
animation_grid(steps)