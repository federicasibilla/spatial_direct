"""
simulate.py: file containing the functions to simulate the death-birth dynamics on the grid
for multiple iteration and store both partial and final results

CONTAINS:
    - simulate: function to update the grid for a given number of steps and store partial and final results
"""

from R_dynamics import *
from N_dynamics import *
from time import time

#----------------------------------------------------------------------------------------------
# defining function to update the grid for a given number of steps
# obs. the first step is different from the others beacuse matrix A needs to be factorized

def simulate(steps,param,mat):
    """
    steps: number of steps to simulate
    param: parameters dictionary
    mat: matrices dictionary

    returns the final N,R and the list of intermediate Ns
    """
    R0 = param['R0']
    N0 = param['N0']

    # first step
    source      = re_soruce(R0,N0,param,mat)
    R_eq,lu,piv = solve_discrete_poisson(source,param)
    g = growth_rates(R_eq,N0,param,mat)
    N_new = encode(death_birth(decode(N0),g))

    # initialize N states list
    states = [decode(N0),decode(N_new)]

    t0 = time()

    # following steps
    for i in range(steps):
        print('step ', i+1)
        # re-compute source based on new equilibrium and death-birth event
        source = re_soruce(R_eq,N_new,param,mat)
        R_eq   = solve_discrete_poisson_2(source,param,lu,piv)
        g      = growth_rates(R_eq,N_new,param,mat)
        N_new  = encode(death_birth(decode(N_new),g))
        states.append(decode(N_new))
    
    t1 = time()
    print(f'time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' seconds')

    return states, R_eq, N_new
