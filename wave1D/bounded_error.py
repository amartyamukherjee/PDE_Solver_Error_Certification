import math

alpha = math.sqrt(0.1)
c = 0.2

if __name__ == "__main__":
    pde_error = 7.19e-5
    init_error = 2.28e-5
    init_error_dx = 8.89e-5
    neumann_error = 3.38e-5
    boundary_error = 1.44e-5
    boundary_error_dt = 2.89e-5
    boundary_error_dt2 = 3.06e-5

    t1 = (neumann_error + boundary_error_dt) / math.sqrt(2*c)
    t2 = (init_error + boundary_error) / math.sqrt(2)
    t3 = init_error_dx * math.sqrt(2*alpha/c)
    t4 = (pde_error + boundary_error_dt2 + c * boundary_error) / math.sqrt(2*c)
    t5 = boundary_error
    print(t1+t2+t3+t4+t5)