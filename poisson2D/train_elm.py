import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

from utils import plot_solutions

# from plot_solutions import plot_solutions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alpha = 0.1

def activation(x):
    return torch.tanh(x)


def activation_prime(x):
    return 1 - torch.tanh(x)**2


def activation_double_prime(x):
    tanh_x = torch.tanh(x)
    return -2 * tanh_x * (1 - tanh_x**2)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(self.hidden.weight, mean=0, std=1)
        nn.init.normal_(self.hidden.bias, mean=0, std=1)
        self.beta = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        return activation(self.hidden(x))


def true_solution(var):
    t = var[..., 0]
    x = var[..., 1]
    return - torch.sin(torch.pi * x) * torch.sin(torch.pi * t) / (2 * torch.pi ** 2)

def source_term(var):
    t = var[..., 0]
    x = var[..., 1]
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * t)

def compute_residuals(net, samples):
    W, b = net.hidden.weight, net.hidden.bias
    x = samples.clone().to(device)
    linear_output = net.hidden(x)  # Linear part of the network
    H = activation(linear_output)  # Apply activation function

    # Pre-allocate memory for derivatives and residuals
    N, m = H.shape
    residuals = torch.zeros((N, m), dtype=torch.float64)

    # H_t_matrix = activation_prime(linear_output) * W[:, 0].unsqueeze(0)
    H_tt_matrix = activation_double_prime(linear_output) * W[:, 0].unsqueeze(0)**2
    H_xx_matrix = activation_double_prime(linear_output) * W[:, 1].unsqueeze(0)**2

    residuals = H_tt_matrix + H_xx_matrix

    b = source_term(x).unsqueeze(1).to(device)
    return residuals, b


def compute_boundary(net, samples):
    x = samples.clone().requires_grad_(True).to(device)
    H = net(x)
    b = 0 * true_solution(x).unsqueeze(1).to(device)
    # print(b.shape)
    return H, b

def compute_neumann(net, samples):
    W, b = net.hidden.weight, net.hidden.bias
    x = samples.clone().requires_grad_(True).to(device)
    # H = net(x)
    linear_output = net.hidden(x)  # Linear part of the network
    H_t_matrix = activation_prime(linear_output) * W[:, 0].unsqueeze(0)
    b = 0 * true_solution(x).unsqueeze(1).to(device)
    # print(b.shape)
    return H_t_matrix, b

def compute_error(net, beta, num_test_points=3000):
    def V(x):
        H = net(x)
        return torch.matmul(H, beta).squeeze(-1)
    x_test = torch.rand((num_test_points, 2), dtype=torch.float64).to(device)
    V_x = V(x_test)
    true_sol = true_solution(x_test)
    error = torch.abs(true_sol - V_x)
    max_error = torch.max(error).item()
    # print("Error: ", error)
    print(f"The maximum error is: {max_error}")
    return max_error


def check_lstsq_residuals(A, b, beta):
    predicted = torch.matmul(A, beta)
    residuals = b - predicted
    max_residual = torch.max(torch.abs(residuals)).item()
    mean_residual = torch.mean(torch.abs(residuals)).item()
    std_residual = torch.std(residuals).item()
    return max_residual, mean_residual, std_residual

RANDOM_SEED = 417
INPUT_DIM = 2
HIDDEN_DIM = 1600
N_SAMPLES = 8000

t_min = 0
t_max = 1
x_min = 0
x_max = 1

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)

    net = Net(INPUT_DIM, HIDDEN_DIM)
    net = net.double().to(device)

    model_path = (f"elm_2d_poisson_{HIDDEN_DIM}_units_{N_SAMPLES}_samples.pt")


    print("Model file does not exist. Formulating and solving ELM problem...")

    samples = torch.rand((N_SAMPLES, INPUT_DIM), dtype=torch.float64) * 2 - 1

    start_time = time.time()
    A0, b0 = compute_residuals(net, samples)
    end_time = time.time()
    solver_time = end_time - start_time
    print(f"Time to compute derivatives: {solver_time} seconds")

    # print("A0:", A0.shape)
    # print("b0:", b0.shape)

    num_ic_points = 100
    num_bc_points = 100

    # t=0 boundary (initial)
    t_ic_points = torch.full((num_ic_points, 1), t_min)
    x_ic_points = (torch.rand(num_ic_points, dtype=torch.float64) * (x_max - x_min) + x_min).unsqueeze(-1)
    ic_points = torch.cat([t_ic_points, x_ic_points], dim=1).double()
    A1, b1 = compute_boundary(net, ic_points)
    # t=1 boundary (initial)
    t_ic1_points = torch.full((num_ic_points, 1), t_max)
    x_ic1_points = (torch.rand(num_ic_points, dtype=torch.float64) * (x_max - x_min) + x_min).unsqueeze(-1)
    ic1_points = torch.cat([t_ic1_points, x_ic1_points], dim=1).double()
    A4, b4 = compute_boundary(net, ic1_points)
    # print("A1:", A1.shape)
    # print("b1:", b1.shape)

    t_bc_points = (torch.rand(num_bc_points, dtype=torch.float64) * (t_max - t_min) + t_min).unsqueeze(-1)
    # x=0 boundary
    x_bc1_points = torch.full((num_bc_points, 1), x_min)
    bc1_points = torch.cat([t_bc_points, x_bc1_points], dim=1).double()
    A2, b2 = compute_boundary(net, bc1_points)
    # x=1 boundary
    x_bc2_points = torch.full((num_bc_points, 1), x_max)
    bc2_points = torch.cat([t_bc_points, x_bc2_points], dim=1).double()
    A3, b3 = compute_boundary(net, bc2_points)

    A = torch.cat([A0, A1, A2, A3, A4])
    b = torch.cat([b0, b1, b2, b3, b4])
    # print("A:", A.shape)
    # print("b:", b.shape)

    start_time = time.time()
    beta = torch.linalg.lstsq(A, b, driver='gels').solution
    # beta = torch.linalg.lstsq(A_reg, b_reg, driver='gelsd').solution
    end_time = time.time()
    solver_time = end_time - start_time
    print(f"Solver time: {solver_time} seconds")

    net.beta.weight.data = beta.view(1, HIDDEN_DIM)

    max_res, mean_res, std_res = check_lstsq_residuals(A, b, beta)

    print(f"Max Residual: {max_res}")
    print(f"Mean Residual: {mean_res}")
    print(f"Standard Deviation of Residuals: {std_res}")

    model_state = {
        "net_state_dict": net.to("cpu").float().state_dict(),
    }
    torch.save(model_state, model_path)
    print(f"Model saved as {model_path}")

    net = net.double().to(device)

    compute_error(net, beta, num_test_points=10000)

    # Call the function to plot the solutions
    plot_solutions(net.to("cpu"), beta.to("cpu"), true_solution)

    # Compute the L2 norms of beta and W
    beta_norm = torch.norm(beta, p=2)
    W = net.hidden.weight
    W_norm = torch.norm(W, p=2)

    # Compute the Lipschitz constant
    lipschitz_constant = beta_norm * W_norm

    print(f"The Lipschitz constant of the network is: {lipschitz_constant:.6f}")