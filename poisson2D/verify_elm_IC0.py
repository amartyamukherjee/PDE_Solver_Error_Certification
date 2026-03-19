import torch
import os
from time import time

from train_elm import Net, INPUT_DIM, HIDDEN_DIM, N_SAMPLES, activation
from verification_utils import verification1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

class Net_Residual_IC(Net):
    def forward(self, x_input):
        # model_output = self.beta(super().forward(x_input))
        model_output = torch.matmul(x_input, self.hidden.weight[:,1:2].t()) + self.hidden.bias[:,None].t()
        model_output = activation(model_output)
        model_output = torch.matmul(model_output, self.beta.weight.t())
        return model_output - 0

if __name__ == "__main__":
    net_residual = Net_Residual_IC(INPUT_DIM, HIDDEN_DIM)
    net_residual = net_residual.to(device)#.double()

    model_path = (f"elm_2d_poisson_{HIDDEN_DIM}_units_{N_SAMPLES}_samples.pt")

    if os.path.exists(model_path):
        print("Model file exists. Loading the model...")
        model_state = torch.load(model_path)
        net_residual.load_state_dict(model_state["net_state_dict"]) # Load state for 'net_residual' as well
        print(net_residual.hidden.weight.data[:,1])
        print(net_residual.hidden.bias.data)
        print(net_residual.beta.weight.data)
        net_residual.eval() # Set to evaluation mode

    samples = (torch.rand((N_SAMPLES, 1), dtype=torch.float32) * 2 - 1).to(device)

    print(net_residual(samples).shape) # Pass 'samples' to the forward method

    num_samples = 100

    x_test = torch.linspace(x_min, x_max, num_samples+1)
    t_test = torch.linspace(t_min, t_max, num_samples+1)

    lb1_list = []
    lb2_list = []
    ub1_list = []
    ub2_list = []

    # start time
    start_time = time()

    # IC
    for xi in range(num_samples):
        # for _ in range(1000):
        # ti = np.random.randint(0, num_samples)
        # xi = np.random.randint(0, num_samples)
        lb1, ub1, lb2, ub2 = verification1D(net_residual,x_test[xi],x_test[xi+1], False, device)
        lb1_list.append(lb1.min().item())
        # lb2_list.append(lb2.min().item())
        lb2_list.append(-1)
        ub1_list.append(ub1.max().item())
        # ub2_list.append(ub2.max().item())
        ub2_list.append(1)
        
        with open("verification_elm_1d_IC_BC_wave.log", "a") as f:
            f.write(f"t: {t_min:.3f}, x: {x_test[xi].item():.3f}, Min lb: {max(min(lb1_list),min(lb2_list))}, Max ub: {min(max(ub1_list),max(ub2_list))}, Time: {time() - start_time:.2f}\n")
        # print("t: ", t_test[ti].item(), ", x:", x_test[xi].item(), ", Min lb: ", max(min(lb1_list),min(lb2_list)), ", Max ub: ", min(max(ub1_list),max(ub2_list)))
    