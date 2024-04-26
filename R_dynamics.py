"""
R_dynamics.py: file to store the implementation of linearization of the
Poisson discrete diffusion problem and the associated functions

CONTAINS:
    - f(R): function to compute the concentration dynamics at a specific grid point
    - re_source: function to calculate the sources matrix for the Poisson problem
    - solve_discrete_poisson: function to solve the discrete Poisson problem with Dirichlet boundary conditions
"""

import numpy as np

from scipy.linalg import lu_factor, lu_solve
from time import time

#-----------------------------------------------------------------------------------
# define f(R(x,y)) where R is the vector containing the concentrations, f the vector
# yielding the concentration dynamics f is the right hand side term of the equation
# ∇^2(R) = -f(R(x,y))

def f(R,R0,N,param,mat):
    """
    R: resources vector (intended at point x,y)
    R0: initial concentration at specific point
    N: species vector (intended at point x,y)
    param: dictionary containing parameters
    mat: dictionary containing matrices

    returns a vector with value of dR/dt at a specific grid point
    """

    # check essential nutrients presence (at each site)
    up_eff = np.ones((mat['uptake'].shape))
    for i in range(len(N)):
        # calculate essential nutrients modulation for each species
        if (np.sum(mat['ess'][i]!=0)):
            mu = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
        else:
            mu = 1
        up_eff[i]=mat['uptake'][i]*mu

    # resource loss due to uptake
    out = np.dot((R*up_eff/(1+R)).T,N.T)
    # resource production due to metabolism
    inn = np.dot(((1/param['w']*(np.dot(param['w']*param['l']*R*up_eff/(1+R),mat['met'].T)))*mat['spec_met']).T,N.T)
    # resource replenishment
    ext = 1/param['tau']*(R0-R)

    return (ext+inn-out)

#-------------------------------------------------------------------------------------------
# re_source: function taking the current concentration and current species matrices, nxnxn_r
# and nxnxn_s respectivley, and the parameters and matrices of the problem to return the 
# resources dynamics matrix, also of shape nxnxn_r, which is the source of Poisson problem

def re_soruce(R,N,param,mat):

    """
    R: current resources concentration matrix nxnxn_R
    N: current species disposition matrix nxnxn_r
    param: parameters dictionary
    mat: matrices dictionary

    returns the nxnxn_r sources matrix 
    """
    source = np.zeros((R.shape[0],R.shape[1],R.shape[2]))

    R0 = param['R0']

    for i in range(R.shape[0]):
        for j in range(R.shape[0]):
            source[i,j,:]=f(R[i,j,:],R0[i,j,:],N[i,j,:],param,mat)

    return source

#-------------------------------------------------------------------------------------------
# solve_discrete_poisson: function taking the parameters dictionary and the source nxnxn_r
# matrix and solving the associated discrete Poisson problem with DBC (fixed to initial value)

def solve_discrete_poisson(source,param):

    """
    source: already discretized source matrix of shape n x n x n_r
    param:  parameters dictionary

    returns the matrix of equilibrium concentrations of shape n x n x n_r; also returns the LU
    factorization of matrix A to use directly for following iterations
    """

    D  = param['D']
    dx = param['L']/param['n']

    # identify mesh shape
    nx  = source.shape[0]
    ny  = source.shape[1]
    n_r = source.shape[2]

    # create 2D mesh based on source matrix dimension
    x = np.linspace(0, 1, source.shape[0])
    y = np.linspace(0, 1, source.shape[1])
    X, Y = np.meshgrid(x, y)

    # initialize n x n x n_r matrix for the solution
    R = np.zeros((nx, ny, n_r))

    # create the matrix A for the linear system AR = b
    A = np.zeros((nx*ny, nx*ny))

    # fill the matrix A to make it matrix laplacian operator with DBC
    for i in range(nx):
        for j in range(ny):
            # linear index corresponding to the point (i, j)
            idx = i*ny + j

            # Dirichlet boundary conditions
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                A[idx, idx] = 1
            # bulk points
            else:
                A[idx, [idx-ny, idx+ny, idx-1, idx+1]] = 1
                A[idx, idx] = -4

    # solve linear system directly by factorization of A (finding A^-1)
    t0 = time()
    print('Factorizing Laplace linear matrix')
    lu, piv = lu_factor(A)

    # solve diffusion problem for each resource separately (factorization is the same)
    for r in range(n_r):

        # create the vector b for the linear system Ax = b
        b = -source[:,:,r].flatten()*dx**2/D

        # dirichelet boundary conditions
        borders = np.zeros((nx, ny), dtype=bool)
        borders[0, :] = borders[-1, :] = borders[:, 0] = borders[:, -1] = True
        borders = borders.ravel()
        b[borders] = param['R0'][:,:,r].ravel()[borders]

        # Solve the linear system using the LU factorization of A
        print(f'solving linear system for resource {r}')
        RR = lu_solve((lu, piv), b)

        # Transform the solution vector into a 2D matrix and store it in the 3D solution matrix
        R[:, :, r] = RR.reshape((nx, ny))

    t1 = time()
    print('time taken to solve for equilibrium: ', round((t1-t0)/60,4), ' seconds')

    return R,lu,piv

#-------------------------------------------------------------------------------------------
# solve_discrete_poisson_2: function taking the parameters dictionary and the source nxnxn_r
# matrix and solving the associated discrete Poisson problem with DBC (fixed to initial value)
# any step after the first one, to avoid re-factorizing

def solve_discrete_poisson_2(source,param,lu,piv):

    """
    source: already discretized source matrix of shape n x n x n_r
    param:  parameters dictionary
    lu,piv: factorization of A matrix computed beforehand

    returns the matrix of equilibrium concentrations of shape n x n x n_r; also returns the LU
    factorization of matrix A to use directly for following iterations
    """

    D  = param['D']
    dx = param['L']/param['n']

    # identify mesh shape
    nx  = source.shape[0]
    ny  = source.shape[1]
    n_r = source.shape[2]

    # create 2D mesh based on source matrix dimension
    x = np.linspace(0, 1, source.shape[0])
    y = np.linspace(0, 1, source.shape[1])
    X, Y = np.meshgrid(x, y)

    # initialize n x n x n_r matrix for the solution
    R = np.zeros((nx, ny, n_r))

    # solve diffusion problem for each resource separately (factorization is the same)
    for r in range(n_r):

        # create the vector b for the linear system Ax = b
        b = -source[:,:,r].flatten()*dx**2/D

        # dirichelet boundary conditions
        borders = np.zeros((nx, ny), dtype=bool)
        borders[0, :] = borders[-1, :] = borders[:, 0] = borders[:, -1] = True
        borders = borders.ravel()
        b[borders] = param['R0'][:,:,r].ravel()[borders]

        # Solve the linear system using the LU factorization of A
        print(f'solving linear system for resource {r}')
        RR = lu_solve((lu, piv), b)

        # Transform the solution vector into a 2D matrix and store it in the 3D solution matrix
        R[:, :, r] = RR.reshape((nx, ny))

    return R