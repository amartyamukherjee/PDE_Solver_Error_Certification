import time 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os

from utils import plot_solutions

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
    t = var[:, 0]
    x = var[:, 1]
    y = var[:, 2]
    z = var[:, 3]
    return torch.exp(-t*3*alpha*(torch.pi**2)) * torch.sin(x*torch.pi) * torch.sin(y*torch.pi) * torch.sin(z*torch.pi)


# def compute_residuals_autograd(net, samples):
#     x = samples.clone().requires_grad_(True)
#     H = net(x)
#     print(H.shape)
#     residuals = []    
#     for i in range(H.shape[1]):
#         # print(i)
#         # process = psutil.Process()
#         # print(f"Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
#         H_ti = torch.autograd.grad(H[:, i], x, grad_outputs=torch.ones_like(H[:, i]), create_graph=True)[0][:, 0] # get first column
#         H_xi = torch.autograd.grad(H[:, i], x, grad_outputs=torch.ones_like(H[:, i]), create_graph=True)[0][:, 1] # get second column
#         H_xxi = torch.autograd.grad(H_xi, x, grad_outputs=torch.ones_like(H_xi), create_graph=True)[0][:, 1] # get second column
#         H_yi = torch.autograd.grad(H[:, i], x, grad_outputs=torch.ones_like(H[:, i]), create_graph=True)[0][:, 2] # get second column
#         H_yyi = torch.autograd.grad(H_yi, x, grad_outputs=torch.ones_like(H_yi), create_graph=True)[0][:, 2] # get second column        
#         residual_per_unit = H_ti - H_xxi - H_yyi
#         residuals.append(residual_per_unit.squeeze())
#     residuals = torch.stack(residuals).t()
#     b = torch.zeros(samples.shape[0], 1)
#     print(residuals.shape)
#     return residuals, b


def compute_residuals(net, samples):
    W, b = net.hidden.weight, net.hidden.bias
    x = samples.clone().to(device)
    linear_output = torch.matmul(x, W.t()) + b  # Linear part of the network
    H = activation(linear_output)  # Apply activation function

    # Pre-allocate memory for derivatives and residuals
    N, m = H.shape
    residuals = torch.zeros((N, m), dtype=torch.float64)  

    H_t_matrix = activation_prime(linear_output) * W[:, 0].unsqueeze(0)
    H_xx_matrix = activation_double_prime(linear_output) * W[:, 1].unsqueeze(0)**2
    H_yy_matrix = activation_double_prime(linear_output) * W[:, 2].unsqueeze(0)**2
    H_zz_matrix = activation_double_prime(linear_output) * W[:, 3].unsqueeze(0)**2

    residuals = H_t_matrix - alpha * H_xx_matrix - alpha * H_yy_matrix - alpha * H_zz_matrix

    b = torch.zeros(N, 1, dtype=torch.float64).to(device)
    return residuals, b

def compute_boundary(net, samples):
    x = samples.clone().requires_grad_(True).to(device)
    H = net(x)
    b = true_solution(x).unsqueeze(1)
    # print(b.shape)    
    return H, b


def compute_error(net, beta, num_test_points=3000):
    def V(x):
        H = net(x)
        return torch.matmul(H, beta).squeeze(-1)
    x_test = torch.rand((num_test_points, 4), dtype=torch.float64).to(device)
    V_x = V(x_test)
    true_sol = true_solution(x_test)
    error = torch.abs(true_sol - V_x)
    max_error = torch.max(error).item()
    # print("Error: ", error)
    print(f"The maximum test error at {num_test_points} points is: "
          f"{max_error}")
    return max_error


def check_lstsq_residuals(A, b, beta):
    predicted = torch.matmul(A, beta)
    residuals = b - predicted
    max_residual = torch.max(torch.abs(residuals)).item()
    mean_residual = torch.mean(torch.abs(residuals)).item()
    std_residual = torch.std(residuals).item()
    return max_residual, mean_residual, std_residual


RANDOM_SEED = 417
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

INPUT_DIM = 4
HIDDEN_DIM = 1600
N_SAMPLES = 30000

t_min = 0
t_max = 1
x_min = 0
x_max = 1
y_min = 0
y_max = 1
z_min = 0
z_max = 1

if __name__ == "__main__":
    net = Net(INPUT_DIM, HIDDEN_DIM)
    net = net.double().to(device)

    model_path = (f"elm_3d_heat_{HIDDEN_DIM}_units_{N_SAMPLES}_samples.pt")


    if os.path.exists(model_path) and False:
        print("Model file exists. Loading the model...")
        model_state = torch.load(model_path)
        net.load_state_dict(model_state["net_state_dict"])
        beta = model_state["beta"]
        net.eval()

    else:
        print("Model file does not exist. Formulating and solving ELM problem...")

        # samples = torch.rand((N_SAMPLES, INPUT_DIM), dtype=torch.float64) * 2 - 1

        samples = torch.rand((N_SAMPLES, INPUT_DIM), dtype=torch.float64)

        start_time = time.time()
        A0, b0 = compute_residuals(net, samples)
        end_time = time.time()
        solver_time = end_time - start_time
        print(f"Time to compute derivatives: {solver_time} seconds")

        # print("A0:", A0.shape)
        # print("b0:", b0.shape)

        num_ic_points = 3000
        num_bc_points = 3000

        # t=0 boundary (initial)
        t_ic_points = torch.full((num_ic_points, 1), t_min)
        x_ic_points = torch.tensor(
            np.random.uniform(x_min, x_max, num_ic_points), dtype=torch.float64
            ).unsqueeze(-1)
        y_ic_points = torch.tensor(
            np.random.uniform(y_min, y_max, num_ic_points), dtype=torch.float64
            ).unsqueeze(-1)
        z_ic_points = torch.tensor(
            np.random.uniform(z_min, z_max, num_ic_points), dtype=torch.float64
            ).unsqueeze(-1)
        ic_points = torch.cat(
            [t_ic_points, x_ic_points, y_ic_points, z_ic_points], dim=1
            ).double()  
        A1, b1 = compute_boundary(net, ic_points)
        # print("A1:", A1.shape)
        # print("b1:", b1.shape)

        # x=0 boundary
        t_bc1_points = torch.tensor(
            np.random.uniform(t_min, t_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        x_bc1_points = torch.full((num_bc_points, 1), x_min)
        y_bc1_points = torch.tensor(
            np.random.uniform(y_min, y_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        z_bc1_points = torch.tensor(
            np.random.uniform(z_min, z_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        bc1_points = torch.cat(
            [t_bc1_points, x_bc1_points, y_bc1_points, z_bc1_points], dim=1
            ).double()  
        A2, b2 = compute_boundary(net, bc1_points)

        # x=1 boundary
        t_bc2_points = torch.tensor(
            np.random.uniform(t_min, t_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        x_bc2_points = torch.full((num_bc_points, 1), x_max)
        y_bc2_points = torch.tensor(
            np.random.uniform(y_min, y_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)   
        z_bc2_points = torch.tensor(
            np.random.uniform(z_min, z_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1) 
        bc2_points = torch.cat(
            [t_bc2_points, x_bc2_points, y_bc2_points, z_bc2_points], dim=1
            ).double()  
        A3, b3 = compute_boundary(net, bc2_points)

        # y=0 boundary
        t_bc3_points = torch.tensor(
            np.random.uniform(t_min, t_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        x_bc3_points = torch.tensor(
            np.random.uniform(x_min, x_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        y_bc3_points = torch.full((num_bc_points, 1), y_min)
        z_bc3_points = torch.tensor(
            np.random.uniform(z_min, z_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1) 
        bc3_points = torch.cat(
            [t_bc3_points, x_bc3_points, y_bc3_points, z_bc3_points], dim=1
            ).double()  
        A4, b4 = compute_boundary(net, bc3_points)

        # y=1 boundary
        t_bc4_points = torch.tensor(
            np.random.uniform(t_min, t_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        x_bc4_points = torch.tensor(
            np.random.uniform(x_min, x_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        y_bc4_points = torch.full((num_bc_points, 1), y_max)
        z_bc4_points = torch.tensor(
            np.random.uniform(z_min, z_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1) 
        bc4_points = torch.cat(
            [t_bc4_points, x_bc4_points, y_bc4_points, z_bc4_points], dim=1
            ).double()  
        A5, b5 = compute_boundary(net, bc4_points)

        # z=0 boundary
        t_bc5_points = torch.tensor(
            np.random.uniform(t_min, t_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        x_bc5_points = torch.tensor(
            np.random.uniform(x_min, x_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        z_bc5_points = torch.full((num_bc_points, 1), z_min)
        y_bc5_points = torch.tensor(
            np.random.uniform(y_min, y_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1) 
        bc5_points = torch.cat(
            [t_bc5_points, x_bc5_points, y_bc5_points, z_bc5_points], dim=1
            ).double()  
        A6, b6 = compute_boundary(net, bc5_points)

        # z=1 boundary
        t_bc6_points = torch.tensor(
            np.random.uniform(t_min, t_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        x_bc6_points = torch.tensor(
            np.random.uniform(x_min, x_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1)
        z_bc6_points = torch.full((num_bc_points, 1), z_max)
        y_bc6_points = torch.tensor(
            np.random.uniform(y_min, y_max, num_bc_points), dtype=torch.float64
            ).unsqueeze(-1) 
        bc6_points = torch.cat(
            [t_bc6_points, x_bc6_points, y_bc6_points, z_bc6_points], dim=1
            ).double()  
        A7, b7 = compute_boundary(net, bc6_points)

        A = torch.cat([A0, A1, A2, A3, A4, A5, A6, A7])
        b = torch.cat([b0, b1, b2, b3, b4, b5, b6, b7])
        print("A:", A.shape)
        print("b:", b.shape)

        def add_regularization(A, lambda_reg, hidden_dim):
            num_cols = A.shape[1]
            reg_matrix = torch.eye(hidden_dim) * lambda_reg
            reg_matrix = reg_matrix.to(device)
            A_reg = torch.cat([A, reg_matrix])
            b_reg = torch.cat([b, torch.zeros(num_cols, 1, dtype=A.dtype).to(device)])
            return A_reg, b_reg

        lambda_reg = 0.0  # Adjust this value based on your needs
        # Add regularization to A
        A_reg, b_reg = add_regularization(A, lambda_reg, HIDDEN_DIM)

        start_time = time.time()
        # beta = torch.linalg.lstsq(A, b, driver='gelsd').solution
        beta = torch.linalg.lstsq(A_reg, b_reg, driver='gels').solution

        # # call scipy lstsq solver
        # A_np = A.cpu().detach().numpy() 
        # b_np = b.cpu().detach().numpy()
        # beta_np, residuals, rank, s = lstsq(A_np, b_np)
        # beta = torch.from_numpy(beta_np).to(dtype=A.dtype, device=A.device)

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
            # "beta": beta
        }
        torch.save(model_state, model_path)
        print(f"Model saved as {model_path}")

        net = net.double().to(device)

    compute_error(net, beta, num_test_points=100000)

    # from matplotlib import rc
    # plt.rcParams['pdf.fonttype'] = 42
    # rc('font', **{'family': 'Linux Libertine'})
    # plt.rcParams['font.size'] = 14
    # plt.rcParams['mathtext.fontset'] = 'stix'



    # Call the function to plot the solutions
    plot_solutions(net.to("cpu"), beta.to("cpu"), true_solution)


    # Compute Lipschits constant of solution 
    def compute_overall_lipschitz_constant(net, beta):
        # Compute the spectral norm of the weight matrix of the hidden layer
        u, s, v = torch.svd(net.hidden.weight.data)
        L_net = torch.max(s).item()
        
        # Compute the l2-norm of beta
        norm_beta = torch.norm(beta, p=2).item()
        
        # The overall Lipschitz constant
        L = L_net * norm_beta
        return L


    L = compute_overall_lipschitz_constant(net, beta)
    print(f"Overall Lipschitz constant of the function V(x): {L}")
