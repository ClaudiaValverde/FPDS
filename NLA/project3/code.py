'''
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import time
from scipy.sparse import csr_matrix, diags


def normalize_graph(G):
    """
    Normalize the graph G to create a column-stochastic matrix.
    """
    n_vec = np.array(G.sum(axis=0)).squeeze()  # Sum of columns (out-degrees)
    np.seterr(divide='ignore')
    Diag = np.where(n_vec != 0, 1 / n_vec, 0)  # Avoid division by zero
    D = diags(Diag)  # Sparse diagonal matrix
    return G @ D     # Normalize G

def handle_dangling_nodes(GD, m):
    """
    Create the teleportation vector z to handle dangling nodes.
    """
    aux = np.array(GD.sum(axis=0)).squeeze()  # Sum of columns of GD
    n = GD.shape[0]
    z = np.where(aux != 0, m / n, 1 / n)  # Adjust for dangling nodes
    return z


def compute_pagerank(G, m=0.85, tol=1e-6, max_iter=100, prin=True):
    """
    Compute the PageRank vector for a graph G using the power iteration method.
    """
    # Step 1: Normalize the graph
    GD = normalize_graph(G)

    # Step 2: Handle dangling nodes
    z = handle_dangling_nodes(GD, m)

    # Step 3: Initialize variables
    n = GD.shape[0]
    x0 = np.ones(n) / n  # Uniform initialization
    norma = 1
    iter_count = 0

    # Step 4: Iterative computation
    while norma > tol and iter_count < max_iter:
        x1 = (1 - m) * GD @ x0 + z @ x0  # PageRank formula
        norma = np.linalg.norm(x1 - x0, np.inf)  # Convergence check
        x0 = x1
        iter_count += 1

    # Step 5: Sort results
    order = np.argsort(-x0)  # Indices sorted by PageRank scores
    pagerank = x0[order]    # Sorted PageRank scores

    # Optional print
    if prin:
        print('Top 10 PageRank indices:', order[:10])
        print('Top 10 PageRank values:', ['{:3.2e}'.format(x) for x in pagerank[:10]])

    return pagerank, order



def compute_pagerank_no_matrix(G, m=0.85, tol=1e-6, max_iter=100, prin=True):
    """
    Compute the PageRank vector without storing the full matrix M.
    """
    n = G.shape[0]  # Number of nodes
    n_vec = np.array(G.sum(axis=0)).squeeze()  # Out-degree of nodes
    dangling_nodes = (n_vec == 0)  # Boolean array for dangling nodes
    out_degree = np.where(n_vec != 0, n_vec, 1)  # Avoid division by zero
    
    #Let us compute the vector L and n_j
    L = []
    n_j = []
    for j in range(0,n):
        #webpages with link to page j
        L_j = G.indices[G.indptr[j]:G.indptr[j+1]]
        L.append(L_j)
        #n_j = length of L_j
        n_j.append(len(L_j))

    # Adjacency list for non-dangling nodes
    adj_list = {j: np.nonzero(G[:, j].toarray().squeeze())[0] for j in range(n)}
    
    # Initialize variables
    x = np.ones(n) / n  # Initial uniform distribution
    x_prev = np.zeros(n)
    iter_count = 0
    
    while np.linalg.norm(x - x_prev, np.inf) > tol and iter_count < max_iter:
        x_prev = x.copy()
        x = np.zeros(n)
        
        # Process each node
        for j in range(n):
            if dangling_nodes[j]:
                x += x_prev[j] / n  # Distribute equally to all nodes
            else:
                for i in adj_list[j]:  # Process non-zero entries
                    x[i] += (1 - m) * x_prev[j] / out_degree[j]
        
        # Add teleportation contribution
        x = x + (m / n)
        iter_count += 1

    # Sort results
    order = np.argsort(-x)  # Indices sorted by PageRank scores
    pagerank = x[order]    # Sorted PageRank scores

    # Optional print
    if prin:
        print('Top 10 PageRank indices:', order[:10])
        print('Top 10 PageRank values:', ['{:3.2e}'.format(x) for x in pagerank[:10]])
    
    return pagerank, order

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import csr_matrix

# Assume compute_pagerank and compute_pagerank_no_matrix are already defined

def compare_pagerank_methods(G, tolerances, m=0.85, max_iter=100):
    """
    Compare results and computational times of PageRank methods with and without matrix storage.
    """
    results_with_matrix = []
    results_no_matrix = []
    times_with_matrix = []
    times_no_matrix = []

    for tol in tolerances:
        # With matrix storage
        start_time = time.time()
        pagerank_with, _ = compute_pagerank(G, m=m, tol=tol, max_iter=max_iter, prin=False)
        times_with_matrix.append(time.time() - start_time)
        results_with_matrix.append(pagerank_with)
        
        # Without matrix storage
        start_time = time.time()
        pagerank_no, _ = compute_pagerank_no_matrix(sp.csc_matrix(G), m=m, tol=tol, max_iter=max_iter, prin=False)
        times_no_matrix.append(time.time() - start_time)
        results_no_matrix.append(pagerank_no)
    
    # Plot results comparison
    plt.figure(figsize=(10, 6))

    # Define a colormap for tolerances
    colormap = plt.cm.get_cmap('viridis', len(tolerances))

    for i, tol in enumerate(tolerances):
        color = colormap(i)  # Pick a distinct color for this tolerance

        # Plot results with matrix storage
        plt.plot(
            results_with_matrix[i],
            label=f'With Matrix, tol={tol:.0e}',
            linestyle='--',
            color=color
        )
        
        # Plot results without matrix storage
        plt.plot(
            results_no_matrix[i],
            label=f'No Matrix, tol={tol:.0e}',
            linestyle='-',
            color=color
        )

    # Add labels, title, and legend
    plt.title('PageRank Results vs. Tolerance')
    plt.xlabel('Node Index')
    plt.ylabel('PageRank Value')
    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small')  # Adjust legend for clarity
    plt.grid(True)
    plt.show()


    # Plot computational time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(tolerances, times_with_matrix, label='With Matrix', marker='o')
    plt.plot(tolerances, times_no_matrix, label='No Matrix', marker='s')
    plt.title('Computational Time vs. Tolerance')
    plt.xscale('log')  # Log scale for tolerances
    plt.gca().invert_xaxis()  # Invert x-axis for decreasing tolerances
    plt.xlabel('Tolerance')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()


if name main:
    
    path_g = "drive/MyDrive/Numerical_Linear_Algebra/project3/p2p-Gnutella30.mtx"

    G = mmread(path_g)

    pagerank_with, order_with = compute_pagerank(G, m=0.15)

    pagerank_without, order_without = compute_pagerank_no_matrix(sp.csc_matrix(G), m=0.15)


    # Example usage
    # Create a random sparse graph for testing
    n = 100  # Number of nodes
    density = 0.05  # Sparsity of the graph
    np.random.seed(42)
    random_graph = csr_matrix(np.random.rand(n, n) < density, dtype=int)

    # Normalize the graph (so it can be used for PageRank computation)
    G = normalize_graph(random_graph)

    # List of tolerances to test
    tolerances = [10**-i for i in range(2, 15, 2)]  # 10^-1 to 10^-12

    # Compare the methods
    compare_pagerank_methods(G, tolerances)
'''

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import time
from scipy.sparse import csr_matrix, diags
import matplotlib.pyplot as plt

def normalize_graph(G):
    """
    Normalize the graph G to create a column-stochastic matrix.
    This ensures that each column of the adjacency matrix sums to 1, which is necessary
    for the power iteration method to converge.

    Args:
        G: Sparse matrix representing the graph.

    Returns:
        A column-stochastic matrix for the graph.
    """
    n_vec = np.array(G.sum(axis=0)).squeeze()  # Sum of columns (out-degrees)
    np.seterr(divide='ignore')  # Ignore division by zero warnings
    Diag = np.where(n_vec != 0, 1 / n_vec, 0)  # Avoid division by zero
    D = diags(Diag)  # Sparse diagonal matrix
    return G @ D     # Normalize G by multiplying with the diagonal matrix

def handle_dangling_nodes(GD, m):
    """
    Create the teleportation vector z to handle dangling nodes.
    Dangling nodes are nodes with no outgoing edges, and they are treated by distributing
    their rank uniformly across all other nodes.

    Args:
        GD: The normalized graph.
        m: Teleportation factor.

    Returns:
        z: Teleportation vector to handle dangling nodes.
    """
    aux = np.array(GD.sum(axis=0)).squeeze()  # Sum of columns of GD
    n = GD.shape[0]  # Number of nodes
    z = np.where(aux != 0, m / n, 1 / n)  # Adjust for dangling nodes
    return z

def compute_pagerank(G, m=0.85, tol=1e-6, max_iter=100, prin=True):
    """
    Compute the PageRank vector for a graph G using the power iteration method.
    The power iteration method repeatedly applies the matrix multiplication until convergence.

    Args:
        G: Sparse matrix representing the graph.
        m: Teleportation factor (usually 0.85).
        tol: Tolerance for convergence (default is 1e-6).
        max_iter: Maximum number of iterations.
        prin: Boolean flag to print the top 10 nodes (default is True).

    Returns:
        pagerank: Computed PageRank vector.
        order: Indices of nodes sorted by PageRank values.
    """
    # Step 1: Normalize the graph
    GD = normalize_graph(G)

    # Step 2: Handle dangling nodes
    z = handle_dangling_nodes(GD, m)

    # Step 3: Initialize variables
    n = GD.shape[0]  # Number of nodes
    x0 = np.ones(n) / n  # Uniform initialization
    norma = 1  # Initial norm for convergence check
    iter_count = 0  # Iteration counter

    # Step 4: Iterative computation (power iteration)
    while norma > tol and iter_count < max_iter:
        x1 = (1 - m) * GD @ x0 + z @ x0  # PageRank formula
        norma = np.linalg.norm(x1 - x0, np.inf)  # Convergence check
        x0 = x1  # Update the rank vector
        iter_count += 1  # Increment iteration count

    # Step 5: Sort results (PageRank scores)
    order = np.argsort(-x0)  # Indices sorted by PageRank scores (descending)
    pagerank = x0[order]    # Sorted PageRank scores

    # Optional print
    if prin:
        print('Top 10 PageRank indices:', order[:10])
        print('Top 10 PageRank values:', ['{:3.2e}'.format(x) for x in pagerank[:10]])

    return pagerank, order

def compute_pagerank_no_matrix(G, m=0.85, tol=1e-6, max_iter=100, prin=True):
    """
    Compute the PageRank vector without storing the full matrix M.
    This method uses adjacency lists and avoids explicitly forming the matrix.

    Args:
        G: Sparse matrix representing the graph.
        m: Teleportation factor.
        tol: Tolerance for convergence (default is 1e-6).
        max_iter: Maximum number of iterations.
        prin: Boolean flag to print the top 10 nodes (default is True).

    Returns:
        pagerank: Computed PageRank vector.
        order: Indices of nodes sorted by PageRank values.
    """
    n = G.shape[0]  # Number of nodes
    n_vec = np.array(G.sum(axis=0)).squeeze()  # Out-degree of nodes
    dangling_nodes = (n_vec == 0)  # Boolean array for dangling nodes
    out_degree = np.where(n_vec != 0, n_vec, 1)  # Avoid division by zero
    
    # Compute the adjacency list for each node
    L = []
    n_j = []
    for j in range(n):
        L_j = G.indices[G.indptr[j]:G.indptr[j+1]]  # Nodes that link to node j
        L.append(L_j)
        n_j.append(len(L_j))  # Number of incoming links

    # Initialize rank vector
    x = np.ones(n) / n  # Initial uniform distribution
    x_prev = np.zeros(n)  # Previous iteration rank
    iter_count = 0  # Iteration counter
    
    while np.linalg.norm(x - x_prev, np.inf) > tol and iter_count < max_iter:
        x_prev = x.copy()  # Store the previous iteration's rank vector
        x = np.zeros(n)  # Reset the rank vector for the current iteration
        
        # Process each node
        for j in range(n):
            if dangling_nodes[j]:
                x += x_prev[j] / n  # Distribute equally to all nodes
            else:
                for i in L[j]:  # Process non-zero entries in the adjacency list
                    x[i] += (1 - m) * x_prev[j] / out_degree[j]
        
        # Add teleportation contribution
        x = x + (m / n)
        iter_count += 1  # Increment iteration count

    # Sort results (PageRank scores)
    order = np.argsort(-x)  # Indices sorted by PageRank scores (descending)
    pagerank = x[order]    # Sorted PageRank scores

    # Optional print
    if prin:
        print('Top 10 PageRank indices:', order[:10])
        print('Top 10 PageRank values:', ['{:3.2e}'.format(x) for x in pagerank[:10]])
    
    return pagerank, order

def compare_pagerank_methods(G, tolerances, m=0.85, max_iter=100):
    """
    Compare results and computational times of PageRank methods with and without matrix storage.
    This function computes the PageRank vectors for different tolerances and compares the results
    and computational times for both methods.

    Args:
        G: Sparse matrix representing the graph.
        tolerances: List of tolerances to use for comparison.
        m: Teleportation factor (default is 0.85).
        max_iter: Maximum number of iterations for the computation.
    """
    results_with_matrix = []
    results_no_matrix = []
    times_with_matrix = []
    times_no_matrix = []

    # Compute PageRank for each tolerance
    for tol in tolerances:
        # With matrix storage
        start_time = time.time()
        pagerank_with, _ = compute_pagerank(G, m=m, tol=tol, max_iter=max_iter, prin=False)
        times_with_matrix.append(time.time() - start_time)
        results_with_matrix.append(pagerank_with)
        
        # Without matrix storage
        start_time = time.time()
        pagerank_no, _ = compute_pagerank_no_matrix(sp.csc_matrix(G), m=m, tol=tol, max_iter=max_iter, prin=False)
        times_no_matrix.append(time.time() - start_time)
        results_no_matrix.append(pagerank_no)
    
    # Plot results comparison
    plt.figure(figsize=(12, 7))
    colormap = plt.cm.get_cmap('viridis', len(tolerances))  # Define colormap

    for i, tol in enumerate(tolerances):
        color = colormap(i)  # Pick a distinct color for this tolerance

        # Plot results with matrix storage
        plt.plot(results_with_matrix[i], label=f'With Matrix, tol={tol:.0e}', linestyle='--', color=color)
        
        # Plot results without matrix storage
        plt.plot(results_no_matrix[i], label=f'No Matrix, tol={tol:.0e}', linestyle='-', color=color)

    plt.title('PageRank Results vs. Tolerance')
    plt.xlabel('Node Index (webpage)')
    plt.ylabel('PageRank Value')
    plt.legend(ncol=2, fontsize='small')  # Adjust legend
    plt.grid(True)
    plt.savefig('2_methods_values.png')
    plt.close()

    # Plot computational time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(tolerances, times_with_matrix, label='With Matrix', marker='o')
    plt.plot(tolerances, times_no_matrix, label='No Matrix', marker='s')
    plt.title('Computational Time vs. Tolerance')
    plt.xscale('log')  # Log scale for tolerances
    plt.gca().invert_xaxis()  # Invert x-axis for decreasing tolerances
    plt.xlabel('Tolerance')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('computation_time.png')
    plt.close()

def compare_pagerank_methods_m(G, m_values, max_iter=100, tol=1e-6):
    """
    Compare results and computational times of PageRank methods with and without matrix storage.
    This function computes the PageRank vectors for different values of m and compares the results
    and computational times for both methods.

    Args:
        G: Sparse matrix representing the graph.
        m_values: List of teleportation factors to use for comparison.
        max_iter: Maximum number of iterations for the computation.
        tol: Tolerance value for convergence (default is 1e-6).
    """
    results_with_matrix = []
    results_no_matrix = []
    times_with_matrix = []
    times_no_matrix = []

    # Compute PageRank for each m value
    for m in m_values:
        # With matrix storage
        start_time = time.time()
        pagerank_with, _ = compute_pagerank(G, m=m, tol=tol, max_iter=max_iter, prin=False)
        times_with_matrix.append(time.time() - start_time)
        results_with_matrix.append(pagerank_with)
        
        # Without matrix storage
        start_time = time.time()
        pagerank_no, _ = compute_pagerank_no_matrix(sp.csc_matrix(G), m=m, tol=tol, max_iter=max_iter, prin=False)
        times_no_matrix.append(time.time() - start_time)
        results_no_matrix.append(pagerank_no)
    
    # Plot results comparison
    plt.figure(figsize=(12, 7))
    colormap = plt.cm.get_cmap('viridis', len(m_values))  # Define colormap

    for i, m in enumerate(m_values):
        color = colormap(i)  # Pick a distinct color for this m value

        # Plot results with matrix storage
        plt.plot(results_with_matrix[i], label=f'With Matrix, m={m:.2f}', linestyle='--', color=color)
        
        # Plot results without matrix storage
        plt.plot(results_no_matrix[i], label=f'No Matrix, m={m:.2f}', linestyle='-', color=color)

    plt.title('PageRank Results vs. Teleportation Factor (m)')
    plt.xlabel('Node Index (webpage)')
    plt.ylabel('PageRank Value')
    plt.legend(ncol=2, fontsize='small')  # Adjust legend
    plt.grid(True)
    plt.savefig('2_methods_values_m.png')
    plt.close()

    # Plot computational time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, times_with_matrix, label='With Matrix', marker='o')
    plt.plot(m_values, times_no_matrix, label='No Matrix', marker='s')
    plt.title('Computational Time vs. Teleportation Factor (m)')
    plt.xlabel('Teleportation Factor (m)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('computation_time_m.png')
    plt.close()

if __name__ == "__main__":
    # Load graph
    path_g = "./p2p-Gnutella30.mtx"
    G = mmread(path_g)

    # Compute PageRank with and without matrix storage
    pagerank_with, order_with = compute_pagerank(G, m=0.15)
    pagerank_without, order_without = compute_pagerank_no_matrix(sp.csc_matrix(G), m=0.15)

    # Example usage
    # Create a random sparse graph for testing
    n = 100  # Number of nodes
    density = 0.05  # Sparsity of the graph
    np.random.seed(42)
    random_graph = csr_matrix(np.random.rand(n, n) < density, dtype=int)

    # Normalize the graph (so it can be used for PageRank computation)
    G = normalize_graph(random_graph)

    # List of tolerances to test
    tolerances = [10**-i for i in range(2, 15, 2)]  # 10^-2 to 10^-14

    # Compare the methods
    compare_pagerank_methods(G, tolerances)

    m_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Compare the methods for different teleportation factors
    compare_pagerank_methods_m(G, m_values)
