import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from R_dynamics import *
from N_dynamics import *
from simulate import *
from visualization import *

# Define the size of the grid
nx, ny, n_r = 50, 50, 3


# Initialize the source matrix with zeros
source = np.zeros((nx, ny, n_r))
# Add a point source in the center of each layer
source[25,25,0]=1
source[15, 5,1]=1
source[45,20,2]=1

# Define the parameters
param = {'R0': np.zeros((nx, ny, n_r)),
         'n' : 50,
         'L' : 50,
         'D' : 1

}

# Call the solve_discrete_poisson function
R = solve_discrete_poisson(source, param)

# Plot the solution for each resource
R_ongrid(R)
R_ongrid_3D(R)