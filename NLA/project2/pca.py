import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Function that computes the 3/4 of the total variance rule
def rule_34(var):
    partial_var = 0
    i=0
    while partial_var < 3/4:
        partial_var += var[i]
        i+=1
    return i
#Function that computes the Kasier rule
def Kasier(S):
    count = 0
    for i in range(len(S)):
        if S[i]**2>1:
            count += 1
    return count

def scree_plot(singular_values, outname, matrix_type=1):
    """
    Create a Scree plot from singular values.
    Parameters:
        singular_values: ndarray
            Singular values from PCA.
        matrix_type: int
            1 for covariance matrix, 2 for correlation matrix.
    """
    eigenvalues = singular_values**2
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', label='Eigenvalues')
    plt.title(f'Scree Plot for the {"Covariance" if matrix_type == 1 else "Correlation"} Matrix')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.legend()
    plt.savefig(outname)
    plt.close()

def pca_with_svd(data, matrix_choice=1):
    """
    Perform PCA using SVD.
    Parameters:
        data: string
            Input dataset (observations x features).
        matrix_choice: int
            1 for covariance matrix, 2 for correlation matrix.
    Returns:
        total_var: ndarray
            Portion of total variance accumulated in each PC.
        std_dev: ndarray
            Standard deviation of each PC.
        pca_coordinates: ndarray
            Expression of the original dataset in PCA coordinates.
        S: matrix
            Singular values for creating Scree plots.
    """

    if data.split('.')[-1] == 'csv':
        # Load the dataset
        data = pd.read_csv(data, index_col=0)
        sample_names = data.index.tolist()

        # Center the data
        X = data.values[:,1:]
    
    else:
        data = np.loadtxt(data)
        sample_names = []

    # Center the data
    X = data - np.mean(data, axis=0)
    n = X.shape[0]

    if matrix_choice == 2:  # Correlation matrix: standardize data
        X = (X / np.std(X, axis=0)).T

    # Perform SVD
    Y = (1 / np.sqrt(n - 1)) * X.T
    U, S, VH = np.linalg.svd(Y, full_matrices=False)

    # Calculate outputs
    total_var = S**2 / np.sum(S**2)  # Proportion of total variance
    std_dev = S  # Standard deviation corresponds to singular values
    pca_coordinates = VH@X#.T  # Expression in PCA coordinates

    return total_var, std_dev, pca_coordinates, sample_names

def pca_with_svd_gene(data_path, matrix_choice=1):
    """
    Perform PCA using SVD.
    Parameters:
        data: string
            Input dataset (observations x features).
        matrix_choice: int
            1 for covariance matrix, 2 for correlation matrix.
    Returns:
        total_var: ndarray
            Portion of total variance accumulated in each PC.
        std_dev: ndarray
            Standard deviation of each PC.
        pca_coordinates: ndarray
            Expression of the original dataset in PCA coordinates.
        singular_values: ndarray
            Singular values for creating Scree plots.
    """

    # Load the dataset
    data = pd.read_csv(data_path)
    sample_names = data.columns[1:]

    # Center the data
    X = data.values[:,1:]

    #Substract the mean
    for i in range(np.shape(X)[0]):
        X[i,] = X[i,]-np.mean(X[i,]) 

    # Center the data
    m = X.shape[0]
    n = X.shape[1]
    print('m, n', m, n)

    if matrix_choice == 2:  # Correlation matrix: standardize data
        X = (X / np.std(X, axis=0))#.T

    # Perform SVD
    Y = 1 / np.sqrt(n-1) * X.T.astype(float)
    U, S, VH = np.linalg.svd(Y, full_matrices=False)

    # Calculate outputs
    total_var = S**2 / np.sum(S**2)  # Proportion of total variance
    std_dev = S  # Standard deviation corresponds to singular values
    pca_coordinates = VH@X#.T  # Expression in PCA coordinates

    return total_var, std_dev, pca_coordinates, sample_names

if __name__=='__main__':
    data_txt = "./data/example.dat"
    data_csv = "./data/RCsGoff.csv"

    '''
    Example.dat datafile
    '''
    print('---'*3, 'Results on example.dat datafile', '---'*3)

    # Apply PCA with SVD (Covariance Matrix)
    total_var_cov, std_dev_cov, pca_coords_cov, sample_names = pca_with_svd(data_txt, matrix_choice=1)

    # Apply PCA with SVD (Correlation Matrix)
    total_var_corr, std_dev_corr, pca_coords_corr, sample_names = pca_with_svd(data_txt, matrix_choice=2)

    print('PCA using Covariance Matrix:')
    print("Total Variance:", total_var_cov)
    print("Standard Deviations:", std_dev_cov)
    print("PCA Coordinates:", pca_coords_cov)
    print('3/4:', rule_34(total_var_cov) )
    print('Kasier:\n', Kasier(std_dev_cov))

    print("\nPCA using Correlation Matrix:")
    print("Total Variance:", total_var_corr)
    print("Standard Deviations:", std_dev_corr)
    print("PCA Coordinates:", pca_coords_corr)
    print('3/4:', rule_34(total_var_corr) )
    print('Kasier:\n', Kasier(std_dev_corr))

    # Create Scree Plots
    scree_plot(std_dev_cov, outname='./scree_exampledat_cov.png', matrix_type=1)
    scree_plot(std_dev_corr, outname='./scree_exampledat_corr.png',  matrix_type=2)

    results_df = pd.DataFrame(data=pca_coords_corr, columns=[f"Obs {i+1}" for i in range(pca_coords_corr.shape[1])])
    # Rename the index to PC1, PC2, PC3, etc.
    results_df.index = [f"PC{i+1}" for i in range(results_df.shape[0])]

    # Save the DataFrame to a CSV file
    results_df.to_csv('./expression_pca_exampledat.csv', index=True)  # Include index in the CSV


    '''
    RCsGoff.csv datafile
    '''
    print('---'*3, 'Results on RCsGoff.csv datafile', '---'*3)

    total_var_cov, std_dev_cov, pca_coords_cov, sample_names = pca_with_svd_gene(data_csv, matrix_choice=1)
    print('PCA using Covariance Matrix:')
    print("Total Variance:", total_var_cov)
    print("Standard Deviations:", std_dev_cov)
    #print("PCA Coordinates:\n", pca_coords_cov)
    print('3/4:', rule_34(total_var_cov) )
    print('Kasier:\n', Kasier(std_dev_cov))
    
    scree_plot(std_dev_cov, outname='./scree_genes.png',  matrix_type=1)

    # saving dataframe of the components
    results_df = pd.DataFrame(data=pca_coords_cov.T, columns=[f"PC{i+1}" for i in range(pca_coords_cov.shape[0])])
    results_df["Variance"] = total_var_cov
    results_df.insert(0, "Sample", sample_names)  # Insert as the first column
    results_df.to_csv('./expression_pca.csv', index=False)

    # PCA plot
    plt.plot(pca_coords_cov[0,],pca_coords_cov[1,],'+k')
    plt.xlabel('PC1: 72.30% variance')
    plt.ylabel('PC2: 15.78% variance')
    plt.savefig('./PCA_plot.png', dpi=300, bbox_inches='tight')
