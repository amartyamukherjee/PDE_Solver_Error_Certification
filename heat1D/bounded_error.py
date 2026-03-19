import math

if __name__ == "__main__":
    pde_error = 2.26e-6
    init_error = 2.16e-4
    boundary_error = 1.81e-8
    boundary_error_dt = 3.73e-8
    boundary_error_0 = 2.13e-6

    t1 = (init_error**2 + 2 * boundary_error_0**2) * 2 * math.e
    t2 = (pde_error + boundary_error_dt + boundary_error) * 2 * math.e
    
    print(math.sqrt(t1+t2))