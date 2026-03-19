import matplotlib.pyplot as plt
import torch

def plot_solutions(net, beta, true_solution, tlim=(0, 1), xlim=(0, 1), resolution=0.01):
    # Generate a grid of points
    t = torch.arange(tlim[0], tlim[1], resolution, dtype=torch.float64)
    x = torch.arange(xlim[0], xlim[1], resolution, dtype=torch.float64)
    T, X = torch.meshgrid(t, x, indexing='ij')
    grid = torch.stack([T.ravel(), X.ravel()], dim=1)
    grid_tensor = grid.to(net.hidden.weight.device) # Ensure grid_tensor is on the same device as net

    # Compute V for the grid
    H = net(grid_tensor)
    V = torch.matmul(H, beta).squeeze(-1).detach()

    # Compute true solution for the grid
    true_sol = true_solution(grid_tensor).detach()

    # Compute error
    error = torch.abs(true_sol - V)
    max_error = torch.max(error).item()

    # Convert tensors to numpy for plotting
    T_np = T.cpu().numpy()
    X_np = X.cpu().numpy()
    V_np = V.cpu().numpy().reshape(X_np.shape)
    true_sol_np = true_sol.cpu().numpy().reshape(X_np.shape)
    error_np = error.cpu().numpy().reshape(X_np.shape)

    # Plotting
    plt.figure(figsize=(18, 6))

    # Plot true solution
    plt.subplot(1, 3, 1)
    plt.contourf(T_np, X_np, true_sol_np, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("True Solution")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot V
    plt.subplot(1, 3, 2)
    plt.contourf(T_np, X_np, V_np, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Predicted Solution")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot error
    plt.subplot(1, 3, 3)
    plt.contourf(T_np, X_np, error_np, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f"Error (Max: {max_error:.2e})")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("solutions_comparison.png")