import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# functions to load the datafiles

def load_data(degree, data):
    data = np.genfromtxt(data, delimiter='  ')#read the data but with two columns of infinite values, we have to remove them
    #print(data)
    points = np.zeros((data.shape[0],2))
    for i in range(0,data.shape[0]):
        points[i,0] = data[i,0]
        points[i,1] = data[i,1]
    #Let us construct the matrix A and the vector b 
    A = np.zeros((points.shape[0],degree))
    for i in range(0,points.shape[0]):
        A[i,:] = [points[i,0]**d for d in range(0,degree)]
    b = points[:,1]
    return A,b,points

def load_csv(data):
    X = np.genfromtxt(data, delimiter=',')
    A, b = X[:,:-1], X[:,-1]
    return A,b

# LS problem solved with SVD
def least_squares_svd(A, b):
    """
    Solve least squares using SVD.
    Optionally, apply Tikhonov regularization for stability.
    """
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=True)
    # Regularized pseudo-inverse of Sigma
    m = U.shape[0]
    n = Vt.shape[0]
    Sigma_inv = np.zeros((n,m))

    for i in np.where(abs(Sigma)>1e-15):
        Sigma_inv[i,i]=1/Sigma[i]

    pseudo_inv = Vt.T @Sigma_inv@U.T
    return pseudo_inv@b

# LS problem solved with QR factorization
def least_squares_qr(A, b, varbose=False):
    """
    Solve least squares using QR decomposition.
    """
    r = np.linalg.matrix_rank(A)
    n = np.shape(A)[1] # Number of columns

    if r == n: # full rank case
      if varbose: print('Full rank matrix')
      Q, R = np.linalg.qr(A, mode = 'complete')
      y = Q.T@b
      y1 = y[:n]                    # First n components of y # compatible part
      R1 = R[:n,:]                  # Leading square part of R
      x = np.linalg.solve(R1, y1)   # Solve triangular system
      y2 = y[n:]                    # orthogonal component
      return x, np.linalg.norm(y2)  # Return solution and residual norm

    else: # rank-defficient case
      if varbose: print('Rank defficient matrix')
      Q, R, P = scipy.linalg.qr(A, pivoting=True) #apply QR with pivoting when its rank defficient
      P = np.identity(len(P))[:, P] # Convert permutation vector to matrix
      y = Q.T@b
      d = y[r:]                     # orthogonal component
      c = y[:r]                     # Leading r components of rotated b
      R1 = R[:r, :r]                # Leading square part of R
      u = np.linalg.solve(R1, c)    # Solve triangular system
      uv = np.zeros(n)              # Initialize solution vector
      uv[:r] = u                    # Fill first r components
      return P@uv, np.linalg.norm(d)**2 # Return permuted solution and residual norm
    
if __name__=="__main__":

    data = "./data/dades.csv"

    print('---'*3, 'Least Square Problem on datafile dades.csv (full-rank)', '---'*3)

    #Run the loop with the new expanded matrix every time

    # Variables
    max_degree = 10
    svd_errors = []
    qr_errors = []
    svd_coefs = []
    qr_coefs = []

    # Loop through degrees
    for d in range(1, max_degree):
        A, b, points = load_data(degree=d, data=data)

        # SVD
        coefficients_svd = least_squares_svd(A, b)
        svd_coefs.append(coefficients_svd)
        svd_errors.append(np.linalg.norm(np.dot(A, coefficients_svd) - b))

        # QR
        coefficients_qr, residual_norm = least_squares_qr(A, b)
        qr_coefs.append(coefficients_qr)
        qr_errors.append(np.linalg.norm(np.dot(A, coefficients_qr) - b))


    '''
    #PLOT THE FITS OF THE LSP
    '''

    # Generate a dense range for plotting the curves
    x_vals = np.linspace(0.5, 8.5, 100)

    # Plotting
    plt.figure(figsize=(6, 4))

    # Plot data points
    plt.scatter(points[:, 0], points[:, 1], color='black', label='Data Points')

    ### SVD plot ###
    for d in range(0, max_degree-1):
        coefs = svd_coefs[d]
        fit = sum(c * x_vals**i for i, c in enumerate(coefs))

        # Plot fit
        plt.plot(x_vals, fit, label=f'Degree {d+1}')

    # Configure plot
    plt.title("Least Squares Problem solved with SVD")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("./graphics/lsp_svd.png")
    plt.close()

    ### QR PLOT ###
    # Generate a dense range for plotting the curves
    x_vals = np.linspace(0.5, 8.5, 100)

    # Plotting
    plt.figure(figsize=(6, 4))

    # Plot data points
    plt.scatter(points[:, 0], points[:, 1], color='black', label='Data Points')

    for d in range(0, max_degree-1):
        coefs = qr_coefs[d]
        fit = sum(c * x_vals**i for i, c in enumerate(coefs))

        # Plot fit
        plt.plot(x_vals, fit, label=f'Degree {d+1}')

    # Configure plot
    plt.title("Least Squares Problem solved with QR")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("./graphics/lsp_qr.png")
    plt.close()

    '''
    Errors plot
    '''
    # Plot the errors for SVD and QR methods
    plt.figure(figsize=(6, 4))
    plt.plot(svd_errors, label='SVD', marker='o', color='blue', linestyle='-')
    plt.plot(qr_errors, label='QR', marker='s', color='red', linestyle='-.')

    # Add title and labels
    plt.title('Error for Each Degree with Both Methods', fontsize=14)
    plt.xlabel('Maximum Degree of the Polynomial Fit', fontsize=12)
    plt.ylabel('Total Error', fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig("./graphics/lsp_error.png")
    plt.close()

    '''
    Identity plot
    '''

    # Flatten the coefficients into single lists for comparison
    svd_flattened = np.concatenate(svd_coefs)
    qr_flattened = np.concatenate(qr_coefs)

    # Scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(svd_flattened, qr_flattened, color='green', alpha=0.7, label='QR vs SVD Coefficients')

    # Identity line (y = x)
    min_val = min(svd_flattened.min(), qr_flattened.min())
    max_val = max(svd_flattened.max(), qr_flattened.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='orange', linestyle='--', label='Identity Line (y=x)')

    # Add labels and title
    plt.title('Comparison of Coefficients from SVD and QR', fontsize=14)
    plt.xlabel('SVD Coefficients', fontsize=12)
    plt.ylabel('QR Coefficients', fontsize=12)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig("./graphics/lsp_id.png")
    plt.close()

    ### Printing some examples ###
    print('Examples of the SVD solution for full-rank:')

    A, b, points = load_data(1, data)
    sol_svd_10 = least_squares_svd(A,b)
    print('Solution norm for degree 1:' , np.linalg.norm(sol_svd_10))
    print('Error:', np.linalg.norm(np.dot(A,sol_svd_10)-b))
    print('\n')

    A, b, points = load_data(2, data)
    sol_svd_10 = least_squares_svd(A,b)
    print('Solution norm for degree 2:' , np.linalg.norm(sol_svd_10))
    print('Error:', np.linalg.norm(np.dot(A,sol_svd_10)-b))
    print('\n')

    A, b, points = load_data(3, data)
    sol_svd_10 = least_squares_svd(A,b)
    print('Solution norm for degree 3:' , np.linalg.norm(sol_svd_10))
    print('Error:', np.linalg.norm(np.dot(A,sol_svd_10)-b))
    print('\n')

    A, b, points = load_data(10, data)
    sol_svd_10 = least_squares_svd(A,b)
    print('Solution norm for degree 10:' , np.linalg.norm(sol_svd_10))
    print('Error:', np.linalg.norm(np.dot(A,sol_svd_10)-b))
    print('\n')

    '''
    Rank Defficient
    '''
    print('---'*3, 'Least Square Problem on datafile dades_regressio.csv (rank deficient)', '---'*3)
    data_r = "./data/dades_regressio.csv"

    A, b = load_csv(data=data_r)

    svd_coefs = []
    svd_errors = []
    qr_coefs = []
    qr_errors = []

    # SVD
    coefficients_svd = least_squares_svd(A, b)
    svd_coefs.append(coefficients_svd)
    svd_errors.append(np.linalg.norm(np.dot(A, coefficients_svd) - b))
    #print('Coefficients svd', coefficients_svd)

    # QR
    coefficients_qr, residual_norm = least_squares_qr(A, b)
    qr_coefs.append(coefficients_qr)
    #print('Coefficients QR', coefficients_qr)
    qr_errors.append(np.linalg.norm(np.dot(A, coefficients_qr) - b))

    sol_svd_10 = least_squares_svd(A,b)
    print('Solution norm for rank defficient:' , np.linalg.norm(sol_svd_10))
    print('Error:', np.linalg.norm(np.dot(A,sol_svd_10)-b))
    print('\n')

    # Flatten the coefficients into single lists for comparison
    svd_flattened = np.concatenate(svd_coefs)
    qr_flattened = np.concatenate(qr_coefs)

    # Scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(svd_flattened, qr_flattened, color='green', alpha=0.7, label='QR vs SVD Coefficients')

    # Identity line (y = x)
    min_val = min(svd_flattened.min(), qr_flattened.min())
    max_val = max(svd_flattened.max(), qr_flattened.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='orange', linestyle='--', label='Identity Line (y=x)')

    # Add labels and title
    plt.title('Comparison of Coefficients from SVD and QR', fontsize=14)
    plt.xlabel('SVD Coefficients', fontsize=12)
    plt.ylabel('QR Coefficients', fontsize=12)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    plt.savefig("./graphics/lsp_id_def.png")
    plt.close()