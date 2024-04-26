"""
N_dynamics.py: a file to contain functions to implement the cellular automata
                     dynamics

CONTAINS:
    - growth_rates: function to calculate growth rates, given the current equilibrium
                    concentration on the grid
    - death_birth: function to simulate one step of death birth process
    - encode: function to one hot encode species matrix
    - decode: function to decode species matrix
"""

import numpy as np
import random

#-----------------------------------------------------------------------------------------
# define growth_rates(R,N,param) function to calculate the growth rates of each individual
# based in their intrinsic conversion factor and on the concentrations of resources on the
# underlying grids of equilibrium concentrations

def growth_rates(R,N,param,mat):
    """
    R: current concentration vectors nxnxn_r
    N: current species vectors nxnxn_s
    param: parameters dictionary
    mat: dictionary of matrices

    returns nxnx1 matrix with growth rates of each individual

    """
    growth_matrix = np.zeros((N.shape[0],N.shape[1]))  # matrix to store growth rates

    up_eff = np.zeros((mat['uptake'].shape))
    n_s = N.shape[2]
    for i in range(n_s):
        if ((R>=mat['ess'][i]).all()):
            up_eff[i]=mat['uptake'][i]
        species_i_matrix = N[:, :, i]                  # level i of N matrix is matrix of species i
        uptake = np.sum((R*up_eff[i,:]*mat['sign'][i,:]*param['w']/(1+R)),axis=2)
        growth_matrix += species_i_matrix*param['g'][i]*(uptake-param['m'][i])
 
    return growth_matrix

#-----------------------------------------------------------------------------------------------
# define death_birth(state,G) the rule of update of a single step of the automaton

def death_birth(state,G):
    """
    state: nxn matrix containing the species pattern, in a not encoded way
    G: nxn matrix containing growth rates at each site

    returns the updated state of the dispsition, where one cell has died and one of its
    neighbours reproducted

    """
    # choose cell to kill
    i = random.randint(0, state.shape[0]-1) 
    j = random.randint(0, state.shape[0]-1)

    # look at the 8 neighbours (numbered from the bottom counterclockwise)and index for pbc
    n1j = n8j = n7j = j-1
    n3j = n4j = n5j = (j+1) % state.shape[0]
    n2j = n6j       = j

    n7i = n6i = n5i = i-1
    n1i = n2i = n3i = (i+1) % state.shape[0]
    n8i = n4i       = i

    # only kill cells close to an interface (keep searching)
    while ((state[i,j]==np.array([state[n1i,n1j],state[n2i,n2j],state[n3i,n3j],state[n4i,n4j],
                   state[n5i,n5j],state[n6i,n6j],state[n7i,n7j],state[n8i,n8j]])).all()):

                   if (state[0,0]==state).all():
                       print('one species has taken up all colony space')
                       return state
                   i = random.randint(0, state.shape[0]-1) 
                   j = random.randint(0, state.shape[0]-1)

                   # look at the 8 neighbours (numbered from the bottom counterclockwise)and index for pbc
                   n1j = n8j = n7j = j-1
                   n3j = n4j = n5j = (j+1) % state.shape[0]
                   n2j = n6j       = j
               
                   n7i = n6i = n5i = i-1
                   n1i = n2i = n3i = (i+1) % state.shape[0]
                   n8i = n4i       = i

    i_s = np.array([n1i,n2i,n3i,n4i,n5i,n6i,n7i,n8i])
    j_s = np.array([n1j,n2j,n3j,n4j,n5j,n6j,n7j,n8j])

    # create probability vector from growth rates vector
    gr = np.array([G[n1i,n1j],G[n2i,n2j],G[n3i,n3j],G[n4i,n4j],
                   G[n5i,n5j],G[n6i,n6j],G[n7i,n7j],G[n8i,n8j]
    ])
    # set to zero probability for negative growth rates
    gr[gr<0] = 0
    if sum(gr)!=0:
        prob = gr/(sum(gr))
    else:
        prob = np.ones((len(gr)))/len(gr)

    # extraction of the winner cell
    winner_idx = np.random.choice(np.arange(len(gr)), p=prob)

    # reproduction
    state[i,j]=state[i_s[winner_idx],j_s[winner_idx]]

    return state

#--------------------------------------------------------------------------------------
# define encoding(N) function to one-hot encode the species matrix

def encode(N):
    """
    N: nxn matrix species (integer values, each integer corresponds to a species)

    returns nxnxn_s matrix where species identity is one-hot-encoded

    """
    species = np.unique(N)
    n_s = len(species)

    one_hot_N = np.zeros((*N.shape, n_s), dtype=int)

    # encoding
    for i, value in enumerate(species):
        one_hot_N[:, :, i] = (N == value).astype(int)

    return one_hot_N

#--------------------------------------------------------------------------------------
# define decoding(N) function to one-hot decode the species matrix

def decode(N):
    """
    N: nxnxn_s matrix containing one hot encoded species

    returns decoded N matrix

    """
    # Estrae l'indice del valore massimo lungo l'asse m (dimensione one-hot)
    decoded_N = np.argmax(N, axis=-1)

    return decoded_N

