import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

def svd_compression_color(image_path, output_dir, ranks, name):
    """
    Compress a color image using SVD for specified ranks.
    
    Parameters:
        image_path (str): Path to the input image file.
        output_dir (str): Directory to save the compressed images.
        ranks (list of int): List of ranks to generate compressed images.
        name: name of the output image
    """
    # Read the image
    image = plt.imread(image_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Frobenius norm for comparison
    fro_norm_original = np.linalg.norm(image)

    frobenius = []
    
    # Process each rank
    for r in ranks:
        # Initialize the compressed image
        compressed_image = np.zeros_like(image, dtype=np.float64)
        
        # Process each color channel
        for channel in range(3):  # 0=Red, 1=Green, 2=Blue
            img_channel = image[:, :, channel]
            
            # Perform SVD
            U, S, Vt = svd(img_channel, full_matrices=False)
            
            # Retain only the top `r` singular values
            U_r = U[:, :r]
            S_r = S[:r]
            Vt_r = Vt[:r, :]
            
            # Reconstruct the channel
            compressed_image[:, :, channel] = np.dot(U_r, np.dot(np.diag(S_r), Vt_r))
        
        # Clip values to ensure they fall within valid range for images
        compressed_image = np.clip(compressed_image, 0, 255)
        
        # Compute the Frobenius norm retained
        fro_norm_compressed = np.linalg.norm(compressed_image)
        fro_norm_retained = (fro_norm_compressed / fro_norm_original) * 100
        
        # Save the compressed image
        output_filename = f"{name}_rank_{r}_retained_{fro_norm_retained:.2f}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plt.imsave(output_path, compressed_image.astype(np.uint8))
        
        print(f"Saved: {output_path} | Frobenius Norm Retained: {fro_norm_retained:.2f}%")
        frobenius.append(fro_norm_retained)

    return frobenius


def plot_frobenius_evolution(ranks, ub_frobenius, beach_frobenius, zara_frobenius):
    """
    Plot the Frobenius norm retained percentages for the three images.
    
    Parameters:
        ranks (list of int): List of ranks used for the approximations.
        ub_frobenius (list of float): Frobenius norms retained for Universitat de Barcelona image.
        beach_frobenius (list of float): Frobenius norms retained for Beach image.
        zara_frobenius (list of float): Frobenius norms retained for Zara image.
    """
    plt.figure(figsize=(6, 4))

    # Plot for Universitat de Barcelona
    plt.plot(ranks, ub_frobenius, marker='o', label='Universitat de Barcelona')

    # Plot for Beach
    plt.plot(ranks, beach_frobenius, marker='o', label='Beach')

    # Plot for Zara
    plt.plot(ranks, zara_frobenius, marker='o', label='Zara')

    # Add labels, title, and legend
    plt.title('Evolution of Frobenius Norm Retention (%) with Rank', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frobenius Norm Retained (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig("frobenius_evolution.png")

if __name__=="__main__":

    # Execute for 3 different images

    # universitat de barcelona
    image_path = "./input_images/universitat_barcelona.jpeg"
    output_dir = './compressed_images'
    ub_frobenius = svd_compression_color(image_path, output_dir, ranks=[5, 10, 50, 100], name='ub')

    # beach
    image_path = "./input_images/beach.jpeg"
    output_dir = './compressed_images'
    beach_frobenius = svd_compression_color(image_path, output_dir, ranks=[5, 10, 50, 100], name='beach')

    # zara
    image_path = "./input_images/zara.jpeg"
    output_dir = './compressed_images'
    zara_frobenius = svd_compression_color(image_path, output_dir, ranks=[5, 10, 50, 100], name='zara')

    # evolution plot
    plot_frobenius_evolution([5, 10, 50, 100], ub_frobenius, beach_frobenius, zara_frobenius)


