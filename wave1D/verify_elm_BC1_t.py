import torch
import os
from time import time

from train_elm import Net, INPUT_DIM, HIDDEN_DIM, N_SAMPLES, alpha, activation, activation_prime, activation_double_prime, true_solution, x_max, x_min, t_max, t_min
from verification_utils import verification1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Net_Residual_BC1(Net):
    def forward(self, t_input):
        # t_input: (N,1)
        Wt = self.hidden.weight[:, 0:1].t()  # (1,m)
        Wt2 = self.hidden.weight[:, 1:2].t()  # (1,m)
        b = self.hidden.bias.view(1, -1)     # (1,m)
        beta = self.beta.weight.view(1, -1)  # (1,m)
        # model_output = self.beta(super().forward(x_input))
        model_output = torch.matmul(t_input, Wt) + b + Wt2
        # model_output = torch.matmul(t_input, Wt) + b
        hprime = activation_prime(model_output)
        dt = (hprime * Wt) @ beta.t()          # (N,1)
        return dt
        # # model_output = self.beta(super().forward(x_input))
        # model_output = activation(model_output)
        # model_output = torch.matmul(model_output, self.beta.weight.t())
        # return model_output - 0

if __name__ == "__main__":
    net_residual_bc1 = Net_Residual_BC1(INPUT_DIM, HIDDEN_DIM)
    net_residual_bc1 = net_residual_bc1.to(device)#.double()

    model_path = (f"elm_1d_wave_{HIDDEN_DIM}_units_{N_SAMPLES}_samples.pt")

    if os.path.exists(model_path):
        print("Model file exists. Loading the model...")
        model_state = torch.load(model_path)
        net_residual_bc1.load_state_dict(model_state["net_state_dict"]) # Load state for 'net_residual_bc1' as well
        net_residual_bc1.eval() # Set to evaluation mode

    samples = (torch.rand((N_SAMPLES, 1), dtype=torch.float32) * 2 - 1).to(device)

    print(net_residual_bc1(samples).shape) # Pass 'samples' to the forward method

    # verification(net_residual,x_min,x_max,t_min,t_max)

    num_samples = 100

    x_test = torch.linspace(x_min, x_max, num_samples+1)
    t_test = torch.linspace(t_min, t_max, num_samples+1)

    lb1_list = []
    lb2_list = []
    ub1_list = []
    ub2_list = []

    # start time
    start_time = time()

    # BC 1
    for ti in range(num_samples):
        # for _ in range(1000):
        # ti = np.random.randint(0, num_samples)
        # xi = np.random.randint(0, num_samples)
        lb1, ub1, lb2, ub2 = verification1D(net_residual_bc1,t_test[ti],t_test[ti+1], False, device)
        lb1_list.append(lb1.min().item())
        # lb2_list.append(lb2.min().item())
        lb2_list.append(-1)
        ub1_list.append(ub1.max().item())
        # ub2_list.append(ub2.max().item())
        ub2_list.append(1)
        
        with open("verification_elm_1d_BC_dt_wave.log", "a") as f:
            f.write(f"t: {t_test[ti].item():.3f}, x: {x_max}, (dt), Min lb: {max(min(lb1_list),min(lb2_list))}, Max ub: {min(max(ub1_list),max(ub2_list))}, Time: {time() - start_time:.2f}\n")
        # print("t: ", t_test[ti].item(), ", x:", x_test[xi].item(), ", Min lb: ", max(min(lb1_list),min(lb2_list)), ", Max ub: ", min(max(ub1_list),max(ub2_list)))
    
