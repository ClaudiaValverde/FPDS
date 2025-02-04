
import time
import random
import numpy as np

np.random.seed(2)
random.seed(2)

'''
declaring UTILS functions

These functions are common for all the different exercises
'''

def Newton_step(lamb0,dlamb,s0,ds):
  '''
  function that performs the 2nd step (Step-size correction substep)
  '''
  alp=1
  idx_lamb0=np.array(np.where(dlamb<0))
  if idx_lamb0.size>0:
    alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))

  idx_s0=np.array(np.where(ds<0))
  if idx_s0.size>0:
    alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
  return alp

def initialize_random_matrices(n):
    '''
    this function initializes the different input matrices for the inequality cases (C2,C3,C4)
    '''
    m = n*2
    N = n+2*m

    # initializing the parameters
    m = 2*n
    x = np.zeros((n))
    lamb = np.ones((m))
    s = np.ones((m))
    z0 = np.concatenate((x, lamb, s))
    G = np.identity(n)
    C = np.concatenate((G, - G), axis = 1)
    d = np.full((m), - 10)
    g = np.random.normal(0, 1, (n))

    return n,m,z0,G,C,d,g

def predictor_substep(MKKT, rhs):
    '''
    definition of the 1st step
    '''
    # Solve the KKT system to find the Newton step δz
    delta_z = np.linalg.solve(MKKT, rhs)
    return delta_z

# step 4
def corrector_substep(MKKT, rhs_corrector):
    # Solve the KKT system with modified right-hand side
    delta_z_corr = np.linalg.solve(MKKT, rhs_corrector)
    return delta_z_corr

# step 6
def update_step(z0, delta_z, alpha):
    # Update z with step size
    return z0 + (0.95 * alpha * delta_z)

def F(x, G, g):
    #this function returns the value of the objective function.
    return 0.5 * np.transpose(x).dot(G).dot(x) + np.transpose(g).dot(x)

def objective_function(x, G, g):
    # definition of the function we want to optimize
    return 0.5 * x @ G @ x + g @ x


# function to read matrices and vectors for C5 and C6

#Function to read an external matrix
def read_matrix(path,n,m,symmetric = False):
    """
    Reads a matrix from the file at 'path' and returns an n x m submatrix.

    Parameters:
    - path (str): Path to the matrix file.
    - n (int): Number of rows in the submatrix.
    - m (int): Number of columns in the submatrix.
    - symmetric (bool): Whether the matrix is symmetric. If True, only reads half the matrix and mirrors it.

    Returns:
    - np.ndarray: The n x m matrix.
    """

    with open(path, "r") as file:
        mat=file.readlines()
    matrix=np.zeros((n,m))
    for line in mat:
        row, column, val=line.strip().split()
        matrix[int(row)-1,int(column)-1]=float(val)
        if symmetric == True:
            matrix[int(column)-1, int(row)-1]=float(val)
    return matrix

def read_vector(path, n):
    """
    Reads a vector from the file at 'path' and returns a vector of specified length.

    Parameters:
    - path (str): Path to the vector file.
    - n (int): Desired length of the vector to read.

    Returns:
    - np.ndarray: The vector of specified length.
    """
    with open(path, "r") as file:
        v = file.readlines()

    vector=np.zeros(n)
    for line in v:
        ind,val=line.strip().split()
        vector[int(ind)-1]=float(val)
    return vector

