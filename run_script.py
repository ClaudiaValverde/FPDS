# CODE OF PROJECT 1 NLA SUBJECT
## CLÀUDIA VALVERDE SANCHEZ

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import ldl, solve_triangular, cholesky, solve_banded

#importing utils
from utils import *

'''
functions for inequality constraints (A=0)

C2, C3 and C4
'''

def compute_residuals(G, g, C, d, x, lambda_, s):
    """
    Compute the residuals for the KKT conditions based on the current values of x, lambda, and s,
    considering only inequality constraints (i.e., with A = 0).

    Parameters:
    - G (np.ndarray): Hessian matrix of shape (n, n).
    - g (np.ndarray): Gradient vector of shape (n,).
    - C (np.ndarray): Inequality constraint matrix of shape (m, n).
    - d (np.ndarray): Inequality constraint vector of shape (m,).
    - x (np.ndarray): Decision variable vector of shape (n,).
    - lambda_ (np.ndarray): Lagrange multipliers for inequality constraints of shape (m,).
    - s (np.ndarray): Slack variables vector of shape (m,).

    Returns:
    - np.ndarray: The residual vector F(z).
    """
    # Residual r_L for the optimality condition: Gx + g - C * lambda = 0
    r_L = G @ x + g - C @ lambda_

    # Residual r_C for the inequality constraint: s + d - C^T * x = 0
    r_C = s + d - C.T @ x

    # Residual r_s for the complementarity condition: s_i * lambda_i = 0 for all i
    r_s = s * lambda_  # Element-wise product

    # Combine residuals into a single vector and negate them for -F(z)
    F_z = np.concatenate((r_L, r_C, r_s))
    return F_z


def calculate_KKT_matrix(G, C, m,n, z):

    # Determine the size of the KKT matrix
    N = n + 2 * m  # Since we only have x, lambda, and s (no gamma)
    M_KKT = np.zeros((N, N))

    # Extract components from z for constructing S and Lambda matrices
    delta_x = z[:n]                 # First n elements of z
    delta_lambda = z[n:n+m]         # Next m elements of z (for Lambda)
    delta_s = z[n+m:]               # Last m elements of z (for S)

    # Create diagonal matrices S and Lambda based on delta_s and delta_lambda
    S_diag = np.diag(delta_s)
    Lambda_diag = np.diag(delta_lambda)

    eq1 = np.concatenate((G, -C, np.zeros((n, m))), axis=1)
    eq2 = np.concatenate((-C.T, np.zeros((m, m)), np.identity(m)), axis=1)
    eq3 = np.concatenate((np.zeros((m, n)), S_diag, Lambda_diag), axis=1)
    KKT_matrix = np.concatenate((eq1, eq2, eq3))

    return KKT_matrix

def main_algorithm(G, C, d, g, m, n, z0, varbose=False):
    """
    Main algorithm for solving the optimization problem with inequality constraints only.

    Parameters:
    - G (np.ndarray): Hessian matrix of shape (n, n).
    - C (np.ndarray): Inequality constraint matrix of shape (m, n).
    - d (np.ndarray): Inequality constraint vector of shape (m,).
    - g (np.ndarray): Gradient vector of shape (n,).
    - m (int): Number of inequality constraints.
    - n (int): Dimension of the decision variable vector x.
    - z0 (np.ndarray): Initial guess for the variables (x, lambda, s).

    Returns:
    - np.ndarray: The solution vector.
    """
    # Define stopping criterion
    epsilon = 1e-16
    max_iter = 100
    e = np.ones((m))

    # Decompose z0 into x, lambda, and s
    x0 = z0[:n]
    lambda0 = z0[n:n+m]
    s0 = z0[n+m:]

    # Construct the KKT matrix for inequality constraints only
    MKKT = calculate_KKT_matrix(G, C, m, n, z0)

    for iteration in range(max_iter):

        # Compute the residuals for the KKT system
        rhs = -compute_residuals(G, g, C, d, x0, lambda0, s0) # -F(z)

        # 1) Predictor substep
        delta_z = predictor_substep(MKKT, rhs)

        # 2) Step-size correction substep
        delta_s = delta_z[n+m:]
        delta_lambda = delta_z[n:n+m]
        alpha = Newton_step(lambda0, delta_lambda, s0, delta_s)

        # 3) Compute mu and sigma for the corrector substep
        mu = np.dot(s0.T, lambda0) / m # defining mu
        s_new = s0 + alpha * delta_s # updating s
        lambda_new = lambda0 + alpha * delta_lambda # updating lambda
        mu_tilde = np.dot(s_new.T, lambda_new) / m
        sigma = (mu_tilde / mu) ** 3

        # 4) Corrector substep
        Ds = np.diag(delta_s)
        Dlambda = np.diag(delta_lambda)
        rhs[n+m:] += - Ds @ Dlambda @ e + sigma * mu * e

        # Solve for delta_z_corr in the corrector step using lianlg function
        delta_z = np.linalg.solve(MKKT, rhs)

        # Decompose delta_z_corr into delta_x, delta_lambda, and delta_s
        delta_x = delta_z[:n]
        delta_lambda = delta_z[n:n+m]
        delta_s = delta_z[n+m:]

        # 5) Step-size for the corrector
        alpha = Newton_step(lambda0, delta_lambda, s0, delta_s)

        # 6) Update substep
        #z0 = update_step(z0, delta_z, alpha)
        z0 = z0 + (alpha * delta_z) * 0.95

        # Update variables for the next iteration
        x0, lambda0, s0 = z0[:n], z0[n:n+m], z0[n+m:]
        # We update the Matrix_KKT
        MKKT = calculate_KKT_matrix(G, C, m, n, z0)

        # get right-hand vector to check the stopping criterion
        rL, rC, rs = rhs[:n], rhs[n:n+m], rhs[n+m:]

        # Check stopping criteria
        if np.linalg.norm(-rL) < epsilon or np.linalg.norm(-rC) < epsilon or np.abs(mu) < epsilon:
            if varbose==True:
                print(f"Convergence achieved at iteration {iteration + 1}.")
            break

    return z0, iteration+1, abs(objective_function(x0,G,g)), -objective_function(-g,G,g), np.linalg.cond(MKKT)

'''
C4 - strategy1
'''

def main_algorithm_strategy1(G, C, d, g, m, n, z0, varbose=False):
    """
    Main algorithm for solving the optimization problem with inequality constraints only.

    Parameters:
    - G (np.ndarray): Hessian matrix of shape (n, n).
    - C (np.ndarray): Inequality constraint matrix of shape (m, n).
    - d (np.ndarray): Inequality constraint vector of shape (m,).
    - g (np.ndarray): Gradient vector of shape (n,).
    - m (int): Number of inequality constraints.
    - n (int): Dimension of the decision variable vector x.
    - z0 (np.ndarray): Initial guess for the variables (x, lambda, s).

    Returns:
    - np.ndarray: The solution vector.
    """
    # Define stopping criterion
    epsilon = 1e-16
    max_iter = 100
    e = np.ones(m)

    # Decompose z0 into x, lambda, and s
    x0 = z0[:n]
    lambda0 = z0[n:n+m]
    s0 = z0[n+m:]

    # Construct the KKT matrix for inequality constraints only
    #modified_MKKT = create_modified_KKT_matrix(G, C, m, n, z0)
    Mkkt = np.zeros((n+m,n+m),dtype = float)
    Mkkt[0:n,0:n] = G
    Mkkt[n:n+m,0:n] = -C.T
    Mkkt[0:n,n:n+m] = -C
    Mkkt[n:n+m,n:n+m] = -np.dot(np.linalg.inv(np.diag(lambda0)),np.diag(s0))
    modified_MKKT = Mkkt

    for iteration in range(max_iter):

        # Compute the residuals for the KKT system
        rhs = compute_residuals(G, g, C, d, x0, lambda0, s0)  # F(z)

        # Isolate δs from the 3rd row of the KKT system
        # get right-hand vector to check the stopping criterion
        r1, r2 = rhs[:n], rhs[n:n+m]
        r3 = rhs[n+m:]  # This is the residual associated with the third row #rs
        rhs = np.append( r1, r2 - r3 / lambda0)

        # LDLT factorization on the modified system
        L, D, perm = ldl(modified_MKKT, check_finite = False)

        # Solve the system using LDLT factorization
        y = solve_triangular(L, -rhs, lower=True, unit_diagonal=True, check_finite = False)
        delta = solve_triangular(D@L.T, y, check_finite = False)

        # Decompose delta into delta_x, delta_lambda, and delta_s
        delta_x = delta[:n]
        delta_lambda = delta[n:n+m]
        # Calculate δs using the equation: δs = Λ^-1 (-r3 - Sdλ)
        delta_s = 1/lambda0*(-r3 - s0 * delta_lambda)

        # 2) Step-size correction substep
        alpha = Newton_step(lambda0, delta_lambda, s0, delta_s)

        # 3) Compute mu and sigma for the corrector substep
        mu = np.dot(s0.T, lambda0) / m  # defining mu
        s_new = s0 + alpha * delta_s  # updating s
        lambda_new = lambda0 + alpha * delta_lambda  # updating lambda
        mu_tilde = np.dot(s_new.T, lambda_new) / m
        sigma = (mu_tilde / mu) ** 3

        # 4) Corrector substep
        rhs_corr = np.zeros(n+m)
        rhs_corr[0:n] = r1
        D_dot = delta_s * delta_lambda
        r3 = r3 + D_dot - sigma * mu * e
        rhs_corr[n:n+m] = r2-r3/lambda0

        # Solve the system using LDLT factorization
        y = solve_triangular(L, -rhs_corr, lower=True, unit_diagonal=True, check_finite = False)
        delta_z = solve_triangular(D@L.T, y, check_finite = False)

        # Decompose delta_z into delta_x, delta_lambda, and delta_s
        delta_x = delta_z[:n]
        delta_lambda_2 = delta_z[n:n+m]
        delta_s_2 = 1/lambda0 * (-r3 - s0 * delta_lambda_2) # Calculate δs using the equation: δs = Λ^-1 (-r3 - Sdλ)
        delta_z2 = np.append(delta_z, delta_s_2)

        # 5) Step-size for the corrector
        alpha_corr = Newton_step(lambda0, delta_lambda_2, s0, delta_s_2)

        # 6) Update substep
        #z0 = update_step(z0, delta, alpha_corr)
        z0 = z0 + 0.95*alpha_corr*delta_z2

        # Check stopping criteria
        if np.linalg.norm(-r1) < epsilon or np.linalg.norm(-r2) < epsilon or np.abs(mu) < epsilon:
            if varbose==True:
                print(f"Convergence achieved at iteration {iteration + 1}.")
            break

        # Update variables for the next iteration
        x0, lambda0, s0 = z0[:n], z0[n:n+m], z0[n+m:] 

        #modified_MKKT = create_modified_KKT_matrix(G, C, m, n, z0)
        for j in range(m):
            modified_MKKT[n+j,n+j] = -s0[j]/lambda0[j]

    return z0, iteration+1, abs(objective_function(x0, G, g)), -objective_function(-g, G, g), np.linalg.cond(modified_MKKT)

'''
C4 - strategy 2
'''
def construct_MKKT_strat2(G, C, m, n, z0):
    lamb = z0[n:n+m]
    s = z0[n+m:]

    S = np.diag(s)
    Lambdas = np.diag(lamb)
    Mat = G + C.dot(np.diag(1 / s * lamb)).dot(np.transpose(C))

    return Mat

def main_algorithm_strategy_2(G, C, d, g, m, n, z0, varbose=False):
    """
    Modified main algorithm for solving the optimization problem with inequality constraints only, using Strategy 2.

    Parameters:
    - G (np.ndarray): Hessian matrix of shape (n, n).
    - C (np.ndarray): Inequality constraint matrix of shape (m, n).
    - d (np.ndarray): Inequality constraint vector of shape (m,).
    - g (np.ndarray): Gradient vector of shape (n,).
    - m (int): Number of inequality constraints.
    - n (int): Dimension of the decision variable vector x.
    - z0 (np.ndarray): Initial guess for the variables (x, lambda, s).

    Returns:
    - np.ndarray: The solution vector.
    """
    epsilon = 1e-16
    max_iter = 100
    e = np.ones(m)


    # Decompose z0 into x, lambda, and s
    x0 = z0[:n]
    lambda0 = z0[n:n+m]
    s0 = z0[n+m:]

    # Construct the KKT matrix for inequality constraints only
    modified_MKKT = construct_MKKT_strat2(G, C, m, n, z0)


    for iteration in range(max_iter):
        # Compute the residuals
        rhs = compute_residuals(G, g, C, d, x0, lambda0, s0)
        r1, r2,r3 = rhs[:n], rhs[n:n+m], rhs[n+m:]

        # Compute the modified matrix Ĝ and vector r̂ using Strategy 2
        S_inv = np.diag(1 / s0)
        Lambda = np.diag(lambda0)
        
        # Compute Ĝ and r̂
        G_hat = G + C @ S_inv @ Lambda @ C.T
        r_hat = -C @ S_inv @ (-r3 + lambda0 * r2)
        
        # Solve for δx using Cholesky factorization on Ĝ
        Cho = cholesky(G_hat)

        y = solve_triangular(Cho,-r1-r_hat,lower=True,check_finite = False)
        delta_x = solve_triangular(Cho.T,y,check_finite = False)

        # Compute δλ and δs based on δx
        delta_lambda = (1/s0)*(-r3+lambda0*r2) - np.diag(lambda0/s0)@C.T@delta_x
        delta_s = -r2+C.T@delta_x

        # 2) Compute step size
        alpha = Newton_step(lambda0, delta_lambda, s0, delta_s)

        # 3) Compute mu and sigma for the corrector substep
        mu = np.dot(s0.T, lambda0) / m  # defining mu
        s_new = s0 + alpha * delta_s  # updating s
        lambda_new = lambda0 + alpha * delta_lambda  # updating lambda
        mu_tilde = np.dot(s_new.T, lambda_new) / m
        sigma = (mu_tilde / mu) ** 3

        # 4) corrector substep
        D_dot = delta_s * delta_lambda
        r3 = r3 + D_dot - sigma * mu * e
        r_hat = -C@S_inv@(-r3+lambda0*r2)
        y = solve_triangular(Cho,-r1-r_hat,lower=True,check_finite = False)
        delta_x2 = solve_triangular(Cho.T,y,check_finite = False)
        delta_lambda_2 = (1/s0)*(-r3+lambda0*r2) - np.diag(lambda0/s0)@C.T@delta_x2
        delta_s_2 = -r2+C.T@delta_x2

        #Obtain new alpha and update z0
        alpha2 = Newton_step(lambda0,delta_lambda_2,s0,delta_s_2)

        # 6) Update substep
        delta_z2 = np.concatenate((delta_x2,delta_lambda_2,delta_s_2))
        z0 = update_step(z0, delta_z2, alpha2)

        # Check stopping criteria
        if np.linalg.norm(-r1) < epsilon or np.linalg.norm(-r2) < epsilon or np.abs(mu) < epsilon:
            if varbose == True:
                print(f"Convergence achieved at iteration {iteration + 1}.")
            break

        # Update variables for the next iteration
        x0, lambda0, s0 = z0[:n], z0[n:n+m], z0[n+m:] 

        modified_MKKT = construct_MKKT_strat2(G, C, m, n, z0)


    return z0, iteration+1, abs(objective_function(x0, G, g)), -objective_function(-g, G, g), np.linalg.cond(modified_MKKT)

'''
C3
Loop over different values of n and measure computation time
'''

def test_different_dimensions(from_x, to_x, interval, algorithm):
    results = []

    for n in range(from_x,to_x,interval):
        m = n*2
        
        n,m,z0,G,C,d,g = initialize_random_matrices(n)

        if algorithm == 'C2':
            # Measure computation time
            start_time = time.time()
            out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm(G, C, d, g, m, n, z0, varbose=False)
            end_time = time.time()
        
        if algorithm == 'C4_strategy1':
            # Measure computation time
            start_time = time.time()
            out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm_strategy1(G, C, d, g, m, n, z0, varbose=False)
            end_time = time.time()
        
        if algorithm == 'C4_strategy2':
            # Measure computation time
            start_time = time.time()
            out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm_strategy_2(G, C, d, g, m, n, z0, varbose=False)
            end_time = time.time()


        computation_time = end_time - start_time
        results.append((n, computation_time, iterations, difference_problem, condition_problem, cond_num))

    return results

'''
C5
'''

def compute_residuals_general(G, g, A, b, C, d, x, gamma, lambda_, s):
    """
    Compute the residuals for the KKT conditions based on the current values of x, gamma, lambda, and s.

    Parameters:
    - G (np.ndarray): Hessian matrix of shape (n, n).
    - g (np.ndarray): Gradient vector of shape (n,).
    - A (np.ndarray): Equality constraint matrix of shape (p, n).
    - b (np.ndarray): Equality constraint vector of shape (p,).
    - C (np.ndarray): Inequality constraint matrix of shape (m, n).
    - d (np.ndarray): Inequality constraint vector of shape (m,).
    - x (np.ndarray): Decision variable vector of shape (n,).
    - gamma (np.ndarray): Lagrange multipliers for equality constraints of shape (p,).
    - lambda_ (np.ndarray): Lagrange multipliers for inequality constraints of shape (m,).
    - s (np.ndarray): Slack variables vector of shape (m,).

    Returns:
    - np.ndarray: The negative residual vector -F(z).
    """
    # Residual r_L for the optimality condition: Gx + g - A*gamma - C*lambda = 0
    r_L = G @ x + g - A @ gamma - C @ lambda_

    # Residual r_A for the equality constraint: b - A^T x = 0
    r_A = b - A.T @ x

    # Residual r_C for the inequality constraint: s + d - C^T x = 0
    r_C = s + d - C.T @ x

    # Residual r_s for the complementarity condition: s_i * lambda_i = 0 for all i
    r_s = s * lambda_  # Element-wise product

    # Combine residuals into a single vector and negate them for -F(z)
    F_z = np.concatenate((r_L, r_A, r_C, r_s))
    return F_z

def Mkkt2fun(G,A,C,m,n,p,z0):
    x0, gamma0, lambda0, s0 = z0[:n], z0[n:n+p], z0[n+p:n+p+m], z0[n+p+m:]

    Mkkt = np.zeros((n+p+2*m,n+p+2*m))
    Mkkt[0:n,0:n] = G
    Mkkt[0:n,n:n+p] = -A
    Mkkt[0:n,n+p:n+p+m] = -C
    Mkkt[n:n+p,0:n] = -A.T
    Mkkt[n+p:n+p+m,0:n] = -C.T
    Mkkt[n+p:n+p+m,n+p+m:n+p+2*m] = np.identity(m)
    Mkkt[n+p+m:n+p+2*m,n+p:n+p+m] = np.diag(s0)
    Mkkt[n+p+m:n+p+2*m,n+p+m:n+p+2*m] = np.diag(lambda0)
    return Mkkt

def main_algorithm_general(G, A, C, d,g,b, m, n, p, z0, varbose=False):
    # defining stopping criterion
    epsilon = 1e-16
    max_iter = 100
    e = np.ones((m))

    delta_lanbda=np.zeros(m)
    delta_s=np.zeros(m)

    # Construct the KKT matrix
    MKKT = Mkkt2fun(G,A,C,m,n,p,z0)

    # Predictor substep
    x0, gamma0, lambda0, s0 = z0[:n], z0[n:n+p], z0[n+p:n+p+m], z0[n+p+m:]

    for iteration in range(max_iter):
        # Compute the right-hand side for the KKT system
        rhs = compute_residuals_general(G, g, A, b, C, d, x0, gamma0, lambda0, s0)

        delta_z = predictor_substep(MKKT, -rhs)

        # Step-size correction substep
        delta_lambda = delta_z[n+p:n+p+m]
        delta_s = delta_z[n+p+m:]
        alpha = Newton_step(lambda0, delta_lambda, s0, delta_s)

        # 3) Compute mu and sigma for the corrector substep
        mu = np.dot(s0.T, lambda0) / m # defining mu
        s_new = s0 + alpha * delta_s # updating s
        lambda_new = lambda0 + alpha * delta_lambda # updating lambda
        mu_tilde = np.dot(s_new.T, lambda_new) / m
        sigma = (mu_tilde / mu) ** 3

        # 4) Corrector substep
        rhs[n+p+m:] = rhs[n+p+m:] + delta_s*delta_lambda - sigma * mu * e

        # Solve for delta_z_corr in the corrector step
        delta_z_corr = corrector_substep(MKKT, -rhs)

        # Decompose delta_z_corr into its parts

        delta_lambda_corr = delta_z_corr[n+p:n+p+m]
        delta_s_corr = delta_z_corr[n+p+m:]

        # Step-size for the corrector
        alpha_corr = Newton_step(lambda0, delta_lambda_corr, s0, delta_s_corr)

        # Update substep
        z0 = update_step(z0, delta_z_corr, alpha_corr)
    
        # Recompute residuals to check convergence
        rL, rA, rC, rs = rhs[:n], rhs[n:n+p], rhs[n+p:n+p+m], rhs[n+p+m:]

        # Check stopping criteria
        if np.linalg.norm(-rL) < epsilon or np.linalg.norm(-rC) < epsilon or np.abs(mu) < epsilon:
            if varbose == True:
                print(f"Convergence achieved at iteration {iteration + 1}.")
            break
        
        # Update MKKT and residuals based on new z0
        # Decompose z0 for the new iteration
        x0, gamma0, lambda0, s0 = z0[:n], z0[n:n+p], z0[n+p:n+p+m], z0[n+p+m:]

        for j in range(m):
            MKKT[n+p+m+j,n+p+j] = s0[j]
            MKKT[n+p+m+j,n+p+m+j] = lambda0[j]

    return z0, iteration+1, abs(objective_function(x0,G,g)), -objective_function(-g,G,g), np.linalg.cond(MKKT)

'''
C6
'''

def compute_residuals_general_ldlt(G,g,A,b,C,d,x0,gamma0,lambda0,s0):

    r1 = G@x0 + g - A@gamma0 - C@lambda0
    r2 = b-A.T@x0
    r3 = s0+d-C.T@x0
    r4 = s0*lambda0
    return np.concatenate((r1,r2,r3,r4))

def main_algorithm_general_ldlt(G, A, C, d, g, b, m, n, p, z0, varbose=False):
    # Stopping criterion
    epsilon = 1e-16
    max_iter = 100
    e = np.ones((m))

    delta_lambda = np.zeros(m)
    delta_s = np.zeros(m)
    
    # Predictor substep initial variables
    x0, gamma0, lambda0, s0 = z0[:n], z0[n:n+p], z0[n+p:n+p+m], z0[n+p+m:]

    Mkkt = np.zeros((n+p+m,n+p+m))
    Mkkt[0:n,0:n] = G
    Mkkt[0:n,n:n+p] = -A
    Mkkt[0:n,n+p:n+p+m] = -C
    Mkkt[n:n+p,0:n] = -A.T
    Mkkt[n+p:n+p+m,0:n] = -C.T
    Mkkt[n+p:n+p+m,n+p:n+p+m] = -np.diag(s0/lambda0)
    MKKT = Mkkt

    for iteration in range(max_iter):
        # Compute residuals (right-hand vector)
        rhs = compute_residuals_general_ldlt(G, g, A, b, C, d, x0, gamma0, lambda0, s0)

        r1, r2, r3, r4 = rhs[:n], rhs[n:n+p], rhs[n+p:n+p+m], rhs[n+p+m:]
        rhs = np.concatenate((r1, r2, r3 - r4 / lambda0))

        # LDLT decomposition of KKT matrix
        L, D, perm = ldl(MKKT, check_finite=False)
        inv_perm = np.argsort(perm)

        # Predictor substep solution using LDLT
        y = solve_triangular(L[perm,:], -rhs[perm], lower=True, unit_diagonal=True, check_finite=False)

        ab = np.zeros((3,n+p+m))
        for j in range(n+p+m-1):
            ab[0,j+1] = D[j,j+1]
            ab[1,j] = D[j,j]
            ab[2,j] = D[j+1,j]
        ab[1,-1] = D[-1,-1]

        prov2 = solve_banded((1,1),ab,y)
        prov3 = solve_triangular(L.T[:,perm],prov2,unit_diagonal = True,check_finite = False)
        prov3 = prov3[inv_perm]
        
        # Extract predictor components
        delta_lambda = prov3[n+p:n+p+m]
        delta_s = 1/lambda0*(-r4-s0*delta_lambda)
        
        # Step-size calculation for predictor
        alpha = Newton_step(lambda0, delta_lambda, s0, delta_s)

        # Update `mu` and `sigma`
        mu = np.dot(s0.T, lambda0) / m
        s_new = s0 + alpha * delta_s
        lambda_new = lambda0 + alpha * delta_lambda
        mu_tilde = np.dot(s_new.T, lambda_new) / m
        sigma = (mu_tilde / mu) ** 3

        # Corrector substep: adjust RHS with new sigma term
        r4 += delta_s * delta_lambda - sigma * mu * e

        rhs_corr = np.concatenate((r1,r2,r3-r4/lambda0))
        
        # Solve corrector substep using LDLT
        y_corr = solve_triangular(L[perm,:], -rhs_corr[perm], lower=True, unit_diagonal=True,check_finite = False)
        prov2 = solve_banded((1,1),ab,y_corr)
        prov3 = solve_triangular(L.T[:,perm],prov2,unit_diagonal = True,lower = False)
        prov3 = prov3[inv_perm]

        # Step-size calculation for corrector
        delta_lambda_corr = prov3[n+p:n+p+m]
        delta_s_corr = 1/lambda0*(-r4-s0*delta_lambda_corr)
        delta_z_corr = np.concatenate((prov3,delta_s_corr))
        alpha_corr = Newton_step(lambda0, delta_lambda_corr, s0, delta_s_corr)

        # Update variables
        z0 = update_step(z0, delta_z_corr, alpha_corr)
        
        if np.linalg.norm(r1) < epsilon or np.linalg.norm(r2) < epsilon or np.linalg.norm(r3)<epsilon  or abs(mu) < epsilon:
            if varbose == True:
                print(f"Convergence achieved at iteration {iteration + 1}.")
            break
        
        # Update KKT matrix and decompose z0 for new iteration
        x0, gamma0, lambda0, s0 = z0[:n], z0[n:n+p], z0[n+p:n+p+m], z0[n+p+m:]
        for k in range(m):
          MKKT[n+p+k,n+p+k] = -s0[k]/lambda0[k]

    # Return solution, number of iterations, objective, and condition number
    return z0, iteration + 1, abs(objective_function(x0, G, g)), -objective_function(-g, G, g), np.linalg.cond(MKKT)



if __name__ == "__main__":
    # creating seed so that the random matrices give the same results in the different times I execute the code
    np.random.seed(42)
    random.seed(42)

    print('---'*20)
    print('---'*8, 'RESULTS C2', '---'*8)
    print('---'*20)
    n = 10

    n,m,z0,G,C,d,g = initialize_random_matrices(n)
    
    print('With n=',n)
    start = time.time()
    out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm(G, C, d, g, m, n, z0, varbose=False)
    end = time.time()
    
    print(f"Execution time: {end-start} seconds")
    print(f"Iterations needed: {iterations}")
    print(f"Difference from the real minimum: {difference_problem:.6f}")
    print(f"Condition problem: {condition_problem:.6f}")
    print(f'Condition num: {cond_num}\n')

    print('---'*20)
    print('---'*8, 'RESULTS C3', '---'*8)
    print('---'*20)

    # Run the test and print results
    results = test_different_dimensions(from_x=10,to_x=500,interval=20, algorithm='C2')
    for n, computation_time, iterations, difference_problem, condition_problem, cond_num in results:
        print(f"Dimension n = {n}:" )
        print(f"Execution time: {computation_time:.4f} seconds")
        print(f"Iterations needed: {iterations}")
        print(f"Difference from the real minimum: {difference_problem:.6f}")
        print(f"Condition problem: {condition_problem:.6f}")
        print(f"Condition number: {cond_num:.6f}")
        print('---'*15, '\n')

    # plotting results

    # Separate data for plotting
    dimensions = [res[0] for res in results]
    computation_times = [res[1] for res in results]
    iterations = [res[2] for res in results]
    differences_from_minimum = [res[3] for res in results]
    condition_numbers = [res[5] for res in results]
    complexity_metric = [n + 2 * (2 * n) for n in dimensions]  # N = n + 2*m where m = 2*n

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns

    # Subplot 1: Complexity Metric vs. Dimension n
    axs[1, 0].plot(dimensions, differences_from_minimum, marker='o', color='b')
    axs[1, 0].set_xlabel("Dimension (n)")
    axs[1, 0].set_ylabel("Difference")
    axs[1, 0].set_title("Difference from Real Minimum vs. Dimension")
    axs[1, 0].grid(True)

    # Subplot 2: Execution Time vs. Dimension n
    axs[0, 0].plot(dimensions, computation_times, marker='o', color='g')
    axs[0, 0].set_xlabel("Dimension (n)")
    axs[0, 0].set_ylabel("Execution Time (seconds)")
    axs[0, 0].set_title("Execution Time vs. Dimension")
    axs[0, 0].grid(True)

    # Subplot 3: Iterations vs. Dimension n
    axs[0, 1].plot(dimensions, iterations, marker='o', color='r')
    axs[0, 1].set_xlabel("Dimension (n)")
    axs[0, 1].set_ylabel("Iterations")
    axs[0, 1].set_title("Iterations Needed for Convergence vs. Dimension")
    axs[0, 1].grid(True)

    # Subplot 4: Condition Number vs. Dimension n
    axs[1, 1].plot(dimensions, condition_numbers, marker='o', color='purple')
    axs[1, 1].set_xlabel("Dimension (n)")
    axs[1, 1].set_ylabel("Condition Number")
    axs[1, 1].set_title("Condition Number vs. Dimension")
    axs[1, 1].grid(True)

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig("C3_analysis.png")

    # Close figure to free memory
    plt.close(fig)

    print('---'*20)
    print('---'*8, 'RESULTS C4', '---'*8)
    print('---'*20)
    print('STRATEGY 1:', '\n')

    # Run the test and print results
    results_strategy_1 = test_different_dimensions(from_x=10,to_x=500,interval=20, algorithm='C4_strategy1')
    n_values_1, times_1, iterations_1, diff_min_1, cond_prob_1, cond_num_1 = zip(*results_strategy_1)
    
    for n, computation_time, iterations, difference_problem, condition_problem, cond_num in results_strategy_1:
        print(f"Dimension n = {n}:" )
        print(f"Execution time: {computation_time:.4f} seconds")
        print(f"Iterations needed: {iterations}")
        print(f"Difference from the real minimum: {difference_problem:.6f}")
        print(f"Condition problem: {condition_problem:.6f}")
        print(f"Condition number: {cond_num:.6f}")
        print('---'*15, '\n')


    print('STRATEGY 2:', '\n')

    # Run the test and print results
    results_strategy_2 = test_different_dimensions(from_x=10,to_x=500,interval=20, algorithm='C4_strategy2')
    for n, computation_time, iterations, difference_problem, condition_problem, cond_num in results_strategy_2:
        print(f"Dimension n = {n}:" )
        print(f"Execution time: {computation_time:.4f} seconds")
        print(f"Iterations needed: {iterations}")
        print(f"Difference from the real minimum: {difference_problem:.6f}")
        print(f"Condition problem: {condition_problem:.6f}")
        print(f"Condition number: {cond_num:.6f}")
        print('---'*15, '\n')


    ## ploting comparison plots

    # Extract data for each metric and strategy
    n_values_2, times_2, iterations_2, diff_min_2, cond_prob_2, cond_num_2 = zip(*results_strategy_2)

    # Create a figure with 4 subplots to compare Strategy 1 and Strategy 2
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Comparison of Algorithm Performance: Strategy 1 vs Strategy 2")

    # Plot Execution Time
    axs[0, 0].plot(n_values_1, times_1, label="LDL^T", color="blue")
    axs[0, 0].plot(n_values_2, times_2, label="Cholesky factorization", color="orange")
    axs[0, 0].set_title("Execution Time")
    axs[0, 0].set_xlabel("Dimension n")
    axs[0, 0].set_ylabel("Time (seconds)")
    axs[0, 0].legend()

    # Plot Iterations Needed
    axs[0, 1].plot(n_values_1, iterations_1, label="LDL^T", color="blue")
    axs[0, 1].plot(n_values_2, iterations_2, label="Cholesky", color="orange")
    axs[0, 1].set_title("Iterations Needed")
    axs[0, 1].set_xlabel("Dimension n")
    axs[0, 1].set_ylabel("Iterations")
    axs[0, 1].legend()

    # Plot Difference from the Real Minimum
    axs[1, 0].plot(n_values_1, diff_min_1, label="LDL^T", color="blue")
    axs[1, 0].plot(n_values_2, diff_min_2, label="Cholesky", color="orange")
    axs[1, 0].set_title("Difference from Real Minimum")
    axs[1, 0].set_xlabel("Dimension n")
    axs[1, 0].set_ylabel("Difference")
    axs[1, 0].legend()

    # Plot Condition Number
    axs[1, 1].plot(n_values_1, cond_num_1, label="LDL^T", color="blue")
    axs[1, 1].plot(n_values_2, cond_num_2, label="Cholesky", color="orange")
    axs[1, 1].set_title("Condition Number")
    axs[1, 1].set_xlabel("Dimension n")
    axs[1, 1].set_ylabel("Condition Number in log scale")
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig("strategy_comparison.png")

    print('---'*20)
    print('---'*8, 'RESULTS C5', '---'*8)
    print('---'*20)
    print('OPTPR 1:', '\n')

    # obtaining paths of matrix and vector files
    path_A = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/A.dad"
    path_C = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/C.dad"
    path_G = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/G_gran.dad"

    path_g = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/g.dad"
    path_b = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/b.dad"
    path_d = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/d.dad"

    # Inital scalar variables
    p = 50
    n = 100

    m = 2*n # Given in the exercise
    N = n + int(p) + 2*m # Dimensuon of KKT Matrix (given in KKT System desc.)

    #Initial condition
    z0=np.zeros(N)
    for i in range(n,N):
        z0[i]=1

    G_matrix = read_matrix(path_G, n, n,symmetric=True)
    A_matrix = read_matrix(path_A, n, p, symmetric=False)
    C_matrix = read_matrix(path_C, n, m, symmetric=False)

    g_vector = read_vector(path_g, n)
    d_vector = read_vector(path_d, m)
    b_vector = read_vector(path_b, p)

    start = time.time()
    out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm_general(G_matrix, A_matrix, C_matrix, d_vector, g_vector, b_vector, m, n, p, z0, varbose=False)
    end = time.time()

    print('With n=', n)
    print(f"Execution time: {end-start} seconds")
    print(f"Iterations needed: {iterations}")
    print(f"Difference from the real minimum: {difference_problem:.6f}")
    print(f"Condition problem: {condition_problem:.6f}")
    print(f'Condition num: {cond_num}\n')

    print('OPTPR 2:', '\n')

    # obtaining paths of matrix and vector files
    path_A = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/A.dad"
    path_C = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/C.dad"
    path_G = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/G_gran.dad"

    path_g = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/g.dad"
    path_b = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/b.dad"
    path_d = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/d.dad"

    # Inital scalar variables
    p = 500
    n = 1000

    m = 2*n # Given in the exercise
    N = n + int(p) + 2*m # Dimensuon of KKT Matrix (given in KKT System desc.)

    #Initial condition
    z0=np.zeros(N)
    for i in range(n,N):
        z0[i]=1

    G_matrix = read_matrix(path_G, n, n,symmetric=True)
    A_matrix = read_matrix(path_A, n, p, symmetric=False)
    C_matrix = read_matrix(path_C, n, m, symmetric=False)

    g_vector = read_vector(path_g, n)
    d_vector = read_vector(path_d, m)
    b_vector = read_vector(path_b, p)

    start = time.time()
    out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm_general(G_matrix, A_matrix, C_matrix, d_vector, g_vector, b_vector, m, n, p, z0, varbose=False)
    end = time.time()

    print('With n=', n)
    print(f"Execution time: {end-start} seconds")
    print(f"Iterations needed: {iterations}")
    print(f"Difference from the real minimum: {difference_problem:.6f}")
    print(f"Condition problem: {condition_problem:.6f}")
    print(f'Condition num: {cond_num}\n')

    print('---'*20)
    print('---'*8, 'RESULTS C6', '---'*8)
    print('---'*20)
    print('OPTPR 1:', '\n')

    # obtaining paths of matrix and vector files
    path_A = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/A.dad"
    path_C = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/C.dad"
    path_G = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/G_gran.dad"

    path_g = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/g.dad"
    path_b = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/b.dad"
    path_d = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr1-20241029/d.dad"

    # Inital scalar variables
    p = 50
    n = 100

    m = 2*n # Given in the exercise
    N = n + int(p) + 2*m # Dimensuon of KKT Matrix (given in KKT System desc.)

    #Initial condition
    z0=np.zeros(N)
    for i in range(n,N):
        z0[i]=1

    G_matrix = read_matrix(path_G, n, n,symmetric=True)
    A_matrix = read_matrix(path_A, n, p, symmetric=False)
    C_matrix = read_matrix(path_C, n, m, symmetric=False)

    g_vector = read_vector(path_g, n)
    d_vector = read_vector(path_d, m)
    b_vector = read_vector(path_b, p)

    start = time.time()
    out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm_general_ldlt(G_matrix, A_matrix, C_matrix, d_vector, g_vector, b_vector, m, n, p, z0, varbose=False)
    end = time.time()

    print('With n=', n)
    print(f"Execution time: {end-start} seconds")
    print(f"Iterations needed: {iterations}")
    print(f"Difference from the real minimum: {difference_problem:.6f}")
    print(f"Condition problem: {condition_problem:.6f}")
    print(f'Condition num: {cond_num}\n')

    print('OPTPR 2:', '\n')

    # obtaining paths of matrix and vector files
    path_A = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/A.dad"
    path_C = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/C.dad"
    path_G = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/G_gran.dad"

    path_g = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/g.dad"
    path_b = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/b.dad"
    path_d = "/Users/claudiavalverde/Documents/DS-master/NLA/project1/optpr2-20241029/d.dad"

    # Inital scalar variables
    p = 500
    n = 1000

    m = 2*n # Given in the exercise
    N = n + int(p) + 2*m # Dimensuon of KKT Matrix (given in KKT System desc.)

    #Initial condition
    z0=np.zeros(N)
    for i in range(n,N):
        z0[i]=1

    G_matrix = read_matrix(path_G, n, n,symmetric=True)
    A_matrix = read_matrix(path_A, n, p, symmetric=False)
    C_matrix = read_matrix(path_C, n, m, symmetric=False)

    g_vector = read_vector(path_g, n)
    d_vector = read_vector(path_d, m)
    b_vector = read_vector(path_b, p)

    start = time.time()
    out_matrix, iterations, difference_problem, condition_problem, cond_num = main_algorithm_general_ldlt(G_matrix, A_matrix, C_matrix, d_vector, g_vector, b_vector, m, n, p, z0, varbose=False)
    end = time.time()

    print('With n=', n)
    print(f"Execution time: {end-start} seconds")
    print(f"Iterations needed: {iterations}")
    print(f"Difference from the real minimum: {difference_problem:.6f}")
    print(f"Condition problem: {condition_problem:.6f}")
    print(f'Condition num: {cond_num}\n')