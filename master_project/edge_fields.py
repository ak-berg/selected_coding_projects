'''
Parallelized solving of the Landau-Lifshitz-Gilbert equation (LLG) for different
- numbers of rotated edge rows Nx
- strengths of the rotating edge field 'B_edge'.

The rotating edge field 'B_edge' is rotated once at the left edge of the rectangular racetrack. 
After the rotation, we turn the edge field off and let the system relaxate and determine the topological charge number to check, 
whether a skyrmion has been successfully created.

The magnetic interaction parameters as well as the racetrack parameters are initialized at the beginning of the script file, 
right below the import-section.
'''
import lib_physics as ps
import numpy as np
from numpy import pi, sqrt
import pandas as pd
from functools import partial
import multiprocessing

# Initializing the parameters of the system.
Nx = 4*30
Ny = 4*7
N = Nx*Ny
xlins = np.linspace(0, Nx, Nx, endpoint=False)
ylins = np.linspace(0, Ny, Ny, endpoint=False)
xcart, ycart = np.meshgrid(xlins, ylins)
grid = xcart + 1j*ycart

ex = ps.ex
ey = ps.ey
ez = ps.ez

meV = 10**(-3)
mu = 3*5.79*10**(-5)
J = 11.6*meV
Bz = 1.5*mu
K = 0.35*meV
Dx = 3.17*meV
Dy = 3.17*meV
gamma = 1
alpha = 0.1

# Initial ferromagnetic (FM) spin configuration
n_init = np.tensordot(np.ones(N), ez, axes=0)

# Set the boundary-, neighbor- and empty-sites
boundaries = ps.idx_boundary_sites(N, Nx)
neighbors = ps.get_neighbors(N, Nx)
idx_zero = ps.idx_zero_sides(N, Nx, boundaries)
size_zero = len(idx_zero)

# Reduction of the number of function arguments via the 'partial' class from the 'functools' library
get_Beff = partial(ps.get_Beff, mu, J, Bz, K, Dx, Dy, N, idx_zero, size_zero, neighbors)
get_derivative = partial(ps.get_derivative, gamma, alpha)

# Setting the sequence of edge field strengths 'B_edge' as well as the different regions, the edge field is applied to. 
Bset = np.arange(21, 31, 1, dtype=int)
B_length = len(Bset)
edge_set = []
idx_edges = np.arange(0+7*Nx, N-7*Nx, Nx, dtype=int)
edge_set.append(idx_edges)
for i in range(1, 5):
    idx_edges = np.concatenate((idx_edges, np.arange(i+7*Nx, i+N-7*Nx, Nx, dtype=int)))
    edge_set.append(idx_edges)

# Initial spin state
n_t = np.copy(n_init)
# Time step
dt = 0.00176
# Initial time value
t = 0
# Rotation frequency of the edge field
v = 1*0.357

# Initial relaxation of the system.
for i in range(10000):
    Beff = get_Beff(n_t)
    K1 = get_derivative(n_t, Beff)

    Beff = get_Beff(n_t + dt*K1)
    K2 = get_derivative(n_t + dt*K1, Beff)

    n_t = n_t + (dt/2)*(K1+K2)

    n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
    t += dt

# Spin configuration after the initial relaxation.
n_t_init = np.copy(n_t)

# Two-dimensional array storing whether a skyrmion has been successfully created for different edge region/field strength combinations. 
results = np.zeros((len(edge_set), B_length), dtype=int)

# Definition of a single simulation.
def simulate(result, index):
    idx_edge = edge_set[index]
    edge_length = len(idx_edge)
    bool_array = np.zeros(B_length, dtype=int)

    for j in range(B_length):
        Bj = Bset[j]
        n_t = np.copy(n_t_init)
        t = 0

        apply_edge_field = partial(ps.apply_edge_field, Bj, v)

        for _ in range(int(np.ceil(2*pi/(v*dt)))):
            Beff = get_Beff(n_t)
            Beff[idx_edge] += apply_edge_field(edge_length, t)
            K1 = get_derivative(n_t, Beff)

            Beff = get_Beff(n_t + dt*K1)
            Beff[idx_edge] += apply_edge_field(edge_length, t + dt)
            K2 = get_derivative(n_t + dt*K1, Beff)

            n_t = n_t + (dt/2)*(K1+K2)

            n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
            t += dt

        for _ in range(10000):
            Beff = get_Beff(n_t)
            K1 = get_derivative(n_t, Beff)

            Beff = get_Beff(n_t + dt*K1)
            K2 = get_derivative(n_t + dt*K1, Beff)

            n_t = n_t + (dt/2)*(K1+K2)

            n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
            t += dt

        if np.abs(ps.get_topological_charge(n_t, n_t[neighbors])) >= 0.9 and np.min(n_t[boundaries][:,2]) > 0:
            bool_array[j] = 1

    result[index] = list(bool_array)

# Parallelization via multiprocessing.
num_edges = len(edge_set)
threads = [None] * num_edges
num_threads = len(threads)
manager = multiprocessing.Manager()
lst = manager.list()
for i in range(num_edges):
    lst.append([0] * B_length)

for i in range(num_threads):
    threads[i] = multiprocessing.Process(target=simulate, args=(lst, i))
    threads[i].start()

for i in range(num_threads):
    threads[i].join()

print(lst)
# Save the results in a DataFrame and generate a .csv-file from it.
df = pd.DataFrame(list(lst), columns=Bset, index=np.arange(1,num_edges+1,dtype=int))
df = df.rename(mapper=lambda x: f"{x} T",axis='columns').rename(mapper=lambda x: f"Nx = {x}",axis='rows')
df.to_csv('creation_matrix.csv')