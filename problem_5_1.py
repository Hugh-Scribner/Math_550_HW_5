import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
import sys
from scipy.integrate import RK45

def main():
    # Problem 1
    test = cahnSolver(402, 1, 50, seed = 100)
    colormap = "viridis"

    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(test[0], test[1], test[2], cmap=colormap, shading="auto")
    fig.colorbar(pcm, ax=ax, label="f")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("t = 50")
    plt.show()

    # Problem 2

def make_ic(seed, eps, N):
   # set up random number generator
    rng = np.random.default_rng(seed)
    u0 = eps * rng.standard_normal(N*N)
    u0 = (u0 - u0.mean()).reshape((N,N))
    return u0

def get_mass(U):
    num_samples = len(U)
    m = np.zeros((num_samples,))
    for i in range(num_samples):
        m[i] = sum(U[i])
    return m

def laplacian_no_flux_2d(N, L):
    """
    Sparse 2D Laplacian (5-point stencil) on a square N-by-N grid of nodes
    spanning a domain of side length L (boundary nodes included), with
    zero-flux (homogeneous Neumann) BCs imposed via mirror ghost points.

    Node ordering: k = i + j*N (x varies fastest).
    Grid spacing:  h = L / (N - 1).

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape (N**2, N**2)
    """
    h = L / (N - 1)

    e = np.ones(N - 1)
    upper = e.copy()
    lower = e.copy()
    upper[0] = 2.0     # row 0:   (-2 u_0 + 2 u_1) / h^2
    lower[-1] = 2.0    # row N-1: (2 u_{N-2} - 2 u_{N-1}) / h^2

    D = spa.diags([lower, -2 * np.ones(N), upper], [-1, 0, 1], format="csr") / h**2
    I = spa.identity(N, format="csr")

    A = spa.kron(I, D) + spa.kron(D, I)
    return A.tocsr()

def cahnSolver(N, l, t_max, seed = 0, t_steps=None, eps = None, save_states = False):
    '''
    This function solves the heat equation using the method of finite differences and a users choice of method

    inputs
    ------
    N : number of internal nodes in one direction in spatial variables
    l : length of one spatial edge
    u0 : initial condition
    f : forcing function
    t_max : time you want to solve the heat equation till
    t_steps : the number of steps you want to take to get to t_max
    method : Short string that describes the method you want to use for time stepping

    outputs
    -------
    sol : 4 item list containing X grid, Y grid, time of each step, and a list of snap shots of u 
    '''
    u = []
    N_trunc= N-2

    # Build mesh and Right hand side
    dx = l/(N-1)
    x = np.linspace(0, l, num = N, endpoint = True)
    y = np.linspace(0, l, num = N, endpoint = True)
    x = x[1:-1] # Truncate off boundaries 
    y = y[1:-1] 
    xx, yy = np.meshgrid(x,y, indexing = 'ij')
    xx_vec, yy_vec = xx.ravel(), yy.ravel() # get grids into a convenient shape for multiple calcs of f

    if eps is None:
        eps = dx

    b = make_ic(seed, eps, N_trunc)

    u.append(b)

    print("Building BE system")
    if t_steps is None: 
        t_steps = int(np.floor(t_max/dx))
        print(t_steps)
    L = laplacian_no_flux_2d(N-2, l-2.0*dx) # take off 2 in each dimension 
    I = spa.identity(L.shape[0], format='coo')
    dt = t_max/t_steps
    A = I - dt*eps**2*L
    A_lu = spla.splu(A.tocsc()) # Claude recommended a banded cholesky solve instead - is that important for this homework?
    u_temp = u[0].ravel()
    print("Solving BE system")
    t = 0
    for i in range(t_steps):
        # z = u_temp + dt * f(xx_vec,yy_vec,(i+1)*dt) # Apply the forcing update to the time step, use tn+1 for BE
        # u_star = A_lu.solve(z)
        u_star = A_lu.solve(u_temp)
        u_temp = u_star * np.exp(dt) / (1 + u_star**2 * (np.exp(2*dt)-1))**.5
        if save_states:
            u.append(u_temp.reshape(N_trunc,N_trunc)) 
        t += dt
    t_vec = np.linspace(0,t, t_steps, endpoint=True)
    sol = [xx,yy,t_vec, u_temp.reshape(N_trunc,N_trunc)]
    print(t)
    return sol

def conservativeCahnSolver(N, l, t_max, seed = 0, t_steps=None, eps = None, save_states = False):
    '''
    This function solves the heat equation using the method of finite differences and a users choice of method

    inputs
    ------
    N : number of internal nodes in one direction in spatial variables
    l : length of one spatial edge
    u0 : initial condition
    f : forcing function
    t_max : time you want to solve the heat equation till
    t_steps : the number of steps you want to take to get to t_max
    method : Short string that describes the method you want to use for time stepping

    outputs
    -------
    sol : 4 item list containing X grid, Y grid, time of each step, and a list of snap shots of u 
    '''
    u = []
    N_trunc= N-2

    # Build mesh and Right hand side
    dx = l/(N-1)
    x = np.linspace(0, l, num = N, endpoint = True)
    y = np.linspace(0, l, num = N, endpoint = True)
    x = x[1:-1] # Truncate off boundaries 
    y = y[1:-1] 
    xx, yy = np.meshgrid(x,y, indexing = 'ij')
    xx_vec, yy_vec = xx.ravel(), yy.ravel() # get grids into a convenient shape for multiple calcs of f

    if eps is None:
        eps = dx

    b = make_ic(seed, eps, N_trunc)

    u.append(b)

    print("Building BE system")
    if t_steps is None: 
        t_steps = int(np.floor(t_max/dx))
        print(t_steps)
    L = laplacian_no_flux_2d(N-2, l-2.0*dx) # take off 2 in each dimension 
    I = spa.identity(L.shape[0], format='coo')
    dt = t_max/t_steps
    A = I - dt*eps**2*L
    A_lu = spla.splu(A.tocsc()) # Claude recommended a banded cholesky solve instead - is that important for this homework?
    u_temp = u[0].ravel()
    print("Solving BE system")
    t = 0
    for i in range(t_steps):
        # z = u_temp + dt * f(xx_vec,yy_vec,(i+1)*dt) # Apply the forcing update to the time step, use tn+1 for BE
        # u_star = A_lu.solve(z)
        u_star = A_lu.solve(u_temp)
        u_star_star = u_star * np.exp(dt) / (1 + u_star**2 * (np.exp(2*dt)-1))**.5
        F = ((u_star_star**2 -1)**2)/4
        u_temp = u_star_star + (sum(b.ravel() - u_star_star))/sum(np.sqrt(F)) * np.sqrt(F)
        if save_states:
            u.append(u_temp.reshape(N_trunc,N_trunc)) 
        t += dt
    t_vec = np.linspace(0,t, t_steps, endpoint=True)
    sol = [xx,yy,t_vec, u_temp.reshape(N_trunc,N_trunc)]
    return sol

if __name__ == "__main__":
    main()