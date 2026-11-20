import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def main():
    # Problem 1
    # Get a picture of the naive solver
    rnseed = 101
    N = 202
    colormap = "viridis"

    # Solve each system once, keeping snapshots
    test = cahnSolver(N, 1, 50, seed=rnseed, save_states=True)
    con_test = conservativeCahnSolver(N, 1, 50, seed=rnseed, save_states=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True,
                             constrained_layout=True)

    panels = [
        (axes[0], test,     "Unmodified Allen-Cahn"),
        (axes[1], con_test, "Modified Allen-Cahn"),
    ]

    for ax, sol, title in panels:
        pcm = ax.pcolormesh(sol[0], sol[1], sol[3][-1], cmap=colormap,
                            shading="auto")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_title(title)

    axes[0].set_ylabel("y")
    fig.colorbar(pcm, ax=axes, label="f", shrink=0.8)
    fig.savefig("problem_1_2_comparison.png", dpi=300, bbox_inches="tight")

    print("You can stop code now")
    #Plot mass over Time
    m_bad = get_mass(test[3])
    m_fix = get_mass(con_test[3])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(test[2], m_bad / N**2, label="Base AC")
    ax.plot(test[2], m_fix / N**2, label="MC AC")
    ax.set_xlabel("t")
    ax.set_ylabel("mean of u")
    ax.legend()
    plt.savefig("mass_plot.png", dpi=300, bbox_inches="tight")
    plt.savefig("mass_plot.png", dpi=300, bbox_inches="tight")

    make_movie(con_test, "conservative_ac.gif")

def make_ic(seed, eps, N):
   # set up random number generator
    rng = np.random.default_rng(seed)
    u0 = eps * rng.standard_normal(N*N)
    u0 = (u0 - u0.mean()).reshape((N,N))
    return u0

def get_mass(U):
    return np.array([np.sum(Ui) for Ui in U])

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
    t_vec = np.linspace(0,t, t_steps + 1)
    if save_states:
        return [xx,yy,t_vec, u]
    sol = [xx,yy,t_vec, u_temp.reshape(N_trunc,N_trunc)]
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
    t_vec = np.linspace(0,t, t_steps + 1)
    if save_states:
        return [xx,yy,t_vec,u]
    sol = [xx,yy,t_vec, u_temp.reshape(N_trunc,N_trunc)]
    return sol

def make_movie(sol, filename="cahn.gif", n_frames=200, fps=20,
               cmap="magma", fixed_scale=True):
    """
    sol : solver output with save_states=True -> [xx, yy, t_vec, snapshots]
    n_frames : number of frames to render (subsampled uniformly in time)
    """
    xx, yy, t_vec, states = sol
    x, y = xx[:, 0], yy[0, :]          # 1D coordinate vectors
    idx = np.linspace(0, len(states) - 1, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    kw = dict(vmin=-1, vmax=1) if fixed_scale else {}
    pcm = ax.pcolormesh(x, y, states[0], cmap=cmap, shading="auto", **kw)
    fig.colorbar(pcm, ax=ax, label="u")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"t = {t_vec[0]:.2f}")

    def update(k):
            i = idx[k]
            pcm.set_array(states[i].ravel())
            if not fixed_scale:
                pcm.set_clim(states[i].min(), states[i].max())
            title.set_text(f"t = {t_vec[i]:.2f}")
            return pcm, title

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)

    if filename.endswith(".mp4"):
        writer = FFMpegWriter(fps=fps)
    else:                               # .gif
        writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()