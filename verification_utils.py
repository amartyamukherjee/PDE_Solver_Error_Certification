import torch

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

def verification(residual_net, x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, verbose=True, device=torch.device("cpu")):
    """Find upper and lower bounds on residuals"""
    # Generate (coarse) testing points
    # NOTE: torch.linspace(0, L, 101) crashes Colab
    separation = 2
    x_test = torch.linspace(x_min, x_max, separation).to(device)#.double()
    y_test = torch.linspace(y_min, y_max, separation).to(device)#.double()
    z_test = torch.linspace(z_min, z_max, separation).to(device)#.double()
    t_test = torch.linspace(t_min, t_max, separation).to(device)#.double()
    tx = torch.meshgrid(t_test, x_test, y_test, z_test)
    image = torch.stack(tx, dim=-1).to(device)
    image.requires_grad_(True)

    # Copy the weights to PINN_residual which outputs residuals to the true solution
    # out = residual_net(image)

    bounded_model = BoundedModule(residual_net, torch.zeros_like(image, requires_grad=True), bound_opts={"conv_mode": "patches"})
    bounded_model.eval()

    # Step 2: define perturbation. Here we use a Linf perturbation on input image.
    eps = x_test[1].item() - x_test[0].item()
    norm = torch.inf
    ptb = PerturbationLpNorm(norm = norm, eps = eps)
    # Input tensor is wrapped in a BoundedTensor object.
    bounded_image = BoundedTensor(image, ptb)#.double()
    # We can use BoundedTensor to get model prediction as usual. Regular forward/backward propagation is unaffected.
    # print('Model prediction:', bounded_model(bounded_image))

    # Step 3: compute bounds using the compute_bounds() method.
    if verbose:
      print('Bounding method: backward (CROWN, DeepPoly)')
    with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
      lb1, ub1 = bounded_model.compute_bounds(x=(bounded_image,), method='CROWN')

    # Auxillary function to print bounds.
    def print_bounds(lb, ub):
        lb = lb.detach().cpu().numpy()
        ub = ub.detach().cpu().numpy()
        print('Domain: [', x_min, x_max, y_min, y_max, t_min, t_max, ']', 'Minimum Lower bound:', lb.min(), ', Maximum Upper bound:', ub.max(), ', Max residual: ', residual_net(image + eps/2).abs().max().item())

    if verbose:
        print_bounds(lb1, ub1)

    # Our library also supports the interval bound propagation (IBP) based bounds,
    # but it produces much looser bounds.
    if verbose:
        print('Bounding method: IBP')
    # with torch.no_grad():
    #   lb2, ub2 = bounded_model.compute_bounds(x=(bounded_image,), method='IBP')

    # if verbose:
    #     print_bounds(lb2, ub2)

    return lb1, ub1, -1, 1 # lb2, ub2

