import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_solutions(net, beta, true_solution, xlim=(0, 1), ylim=(0, 1), 
                   resolution=0.01):
    # Generate a grid of points
    t = 1.0
    z = 0.5
    x = np.arange(xlim[0], xlim[1], resolution)
    y = np.arange(ylim[0], ylim[1], resolution)   
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t)
    Z = np.full_like(X, z)
    grid = np.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    grid_tensor = torch.tensor(grid, dtype=torch.float64)

    # Compute V for the grid
    H = net(grid_tensor)
    V = torch.matmul(H, beta).squeeze(-1).detach().numpy().reshape(X.shape)

    # Compute true solution for the grid
    true_sol = true_solution(grid_tensor).detach().numpy().reshape(X.shape)

    # Compute error
    error = np.abs(true_sol - V)
    max_error = np.max(error)

    # Plotting
    plt.figure(figsize=(18, 6))

    # Plot true solution
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, true_sol, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f"True Solution at $t$={t} and $z$={z}")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    # Plot V
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, V, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f"Predicted Solution at $t$={t} and $z$={z}")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    # Plot error
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, error, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f"Error at $t$={t} (Max: {max_error:.2e})")
    plt.title(f"Error at $t$={t} and $z$={z}")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.show()
    plt.savefig("solution_plots.png")
