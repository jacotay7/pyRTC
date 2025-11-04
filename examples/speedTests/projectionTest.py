import numpy as np

# Define Gaussian basis functions
def gaussian(x, center, sigma):
    return np.exp(-np.linalg.norm(x - center)**2 / (2 * sigma**2))

# Define the function you want to approximate (as an example)
def target_function(x):
    return np.sin(np.linalg.norm(x))

# Define a grid of points on the disk
def generate_disk_grid(radius, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = np.linspace(0, radius, num_points)
    r_grid, theta_grid = np.meshgrid(r, theta)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    return np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Parameters
radius = 1.0  # Radius of the disk
num_points = 100  # Number of points on the disk
sigma = 0.2  # Standard deviation for Gaussian profiles
num_gaussians = 100  # Number of Gaussian influence profiles

# Generate the grid on the disk
x_points = generate_disk_grid(radius, num_points)

# Generate random centers for the Gaussians (these could be any predefined locations)
np.random.seed(42)  # For reproducibility
centers = np.random.uniform(low=-radius, high=radius, size=(num_gaussians, 2))

# Create the design matrix Phi
Phi = np.zeros((len(x_points), num_gaussians))
for i, center in enumerate(centers):
    for j, x in enumerate(x_points):
        Phi[j, i] = gaussian(x, center, sigma)

# Evaluate the target function at the grid points
f = np.array([target_function(x) for x in x_points])

# Solve for the weights using the normal equation
w = (np.linalg.inv(Phi.T @ Phi) @ Phi.T ) @ f

# Approximate the function using the optimized weights
approx_values = Phi @ w

# Example of comparing the original function and approximation
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(x_points[:, 0], x_points[:, 1], c=f, label='Original Function', cmap='viridis')
plt.colorbar()
plt.title('Original Function')

plt.figure()
plt.scatter(x_points[:, 0], x_points[:, 1], c=approx_values, label='Approximated Function', cmap='viridis')
plt.colorbar()
plt.title('Approximated Function')
plt.show()
