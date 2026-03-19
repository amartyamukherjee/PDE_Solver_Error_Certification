import math

if __name__ == "__main__":
    pde_error = 3.16e-3
    init_error = 2.90e-4
    boundary_error = 1.83e-4
    
    print(max(init_error,boundary_error)+pde_error)