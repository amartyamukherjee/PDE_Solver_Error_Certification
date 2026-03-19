import torch
import os
from time import time

from train_elm import Net, INPUT_DIM, HIDDEN_DIM, N_SAMPLES, alpha, activation, activation_prime, activation_double_prime, true_solution, x_max, x_min, t_max, t_min
from verification_utils import verification2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Net_PDE_Residual(Net):
    def forward(self, x_input): # Renamed to avoid confusion with global 'x'

        # W = self.hidden.weight
        linear_output = self.hidden(x_input)  # Use the input 'x_input' and self.hidden

        # model_output = torch.matmul(x_input, self.hidden.weight.t()) + self.hidden.bias[:,None].t()
        
        model_output = activation(linear_output)
        model_output = torch.matmul(model_output, self.beta.weight.t())

        # print(model_output - true_solution(x_input))

        true_sol = true_solution(x_input)

        # print("Model output shape: ", model_output.shape, "True solution shape: ", true_sol.shape)
        
        # After loading, self.beta.weight should have shape (HIDDEN_DIM, 1)
        # This computes the weighted residual (residuals @ beta)
        return (model_output - true_sol)**2
    
if __name__ == "__main__":
    net_residual = Net_PDE_Residual(INPUT_DIM, HIDDEN_DIM)
    net_residual = net_residual.to(device)#.double()

    model_path = (f"elm_1d_wave_{HIDDEN_DIM}_units_{N_SAMPLES}_samples.pt")

    if os.path.exists(model_path):
        print("Model file exists. Loading the model...")
        model_state = torch.load(model_path)
        net_residual.load_state_dict(model_state["net_state_dict"]) # Load state for 'net_residual' as well
        net_residual.eval() # Set to evaluation mode

    samples = (torch.rand((N_SAMPLES, INPUT_DIM), dtype=torch.float32) * 2 - 1).to(device)

    print(net_residual(samples)) # Pass 'samples' to the forward method

    # verification(net_residual,x_min,x_max,t_min,t_max)

    num_samples = 1000

    x_test = torch.linspace(x_min, x_max, num_samples+1)
    t_test = torch.linspace(t_min, t_max, num_samples+1)

    lb1_list = []
    lb2_list = []
    ub1_list = []
    ub2_list = []

    # start time
    start_time = time()

    total_ub = 0

    for ti in range(num_samples):
        for xi in range(num_samples):
            # for _ in range(1000):
            # ti = np.random.randint(0, num_samples)
            # xi = np.random.randint(0, num_samples)
            lb1, ub1, lb2, ub2 = verification2D(net_residual,x_test[xi],x_test[xi+1],t_test[ti],t_test[ti+1], False, device)
            lb1_list.append(lb1.min().item())
            # lb2_list.append(lb2.min().item())
            lb2_list.append(-1)
            ub1_list.append(ub1.max().item())
            total_ub += ub1.max().item()/(num_samples**2)
            # ub2_list.append(ub2.max().item())
            ub2_list.append(1)
            
            with open("verification_elm_sol.log", "a") as f:
                f.write(f"t: {t_test[ti].item():.3f}, x: {x_test[xi].item():.3f}, Min lb: {max(min(lb1_list),min(lb2_list))}, Max ub: {total_ub}, Time: {time() - start_time:.2f}\n")
            # print("t: ", t_test[ti].item(), ", x:", x_test[xi].item(), ", Min lb: ", max(min(lb1_list),min(lb2_list)), ", Max ub: ", min(max(ub1_list),max(ub2_list)))
