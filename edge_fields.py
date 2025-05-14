import numpy as np
from numpy import sin, cos, pi, exp, sqrt
import functools
import matplotlib.pyplot as plt
import random
import time
import multiprocessing

Nx = 4*30
Ny = 4*7
N = Nx*Ny
xlins = np.linspace(0, Nx, Nx, endpoint=False)
ylins = np.linspace(0, Ny, Ny, endpoint=False)
xcart, ycart = np.meshgrid(xlins, ylins)
grid = xcart + 1j*ycart

ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])

meV = 10**(-3)
mu = 3*5.79*10**(-5)
J = 11.6*meV
Bz = 0*1.5*mu
K = (0.8+0.35)*meV
Dx = 3.17*meV
Dy = 3.17*meV

n_init = np.tensordot(np.ones(N), ez, axes=0)

#Antiferromagnetic boundaries
#for i in range(1, N):
#    n_init[i] = (-n_init[i])**(i%Nx + int(i/Nx))

def idx_neighbor_sides(num):
    return [(num+1)%N, num-1, (num+Nx)%N, num-Nx]

def idx_boundary_sides():

    bound_array = np.arange(0, Nx, 1)
    bound_array = np.append(bound_array, np.arange(N-Nx, N, 1))
    bound_array = np.append(bound_array, np.arange(Nx, N-Nx+1, Nx))
    bound_array = np.append(bound_array, np.arange(Nx-1, N-Nx, Nx))

    return bound_array

def idx_zero_sides(grid_size, boundaries):
    #idx_zero = np.zeros((len(boundaries), (1, 1))
    idx_zero = []

    for i in boundaries:
        if i < Nx:
            idx_zero.append((i,3))
        elif N-Nx <= i < N:
            idx_zero.append((i,2))
        if i%Nx == 0:
            idx_zero.append((i,1))
        elif (i+1)%Nx == 0:
            idx_zero.append((i,0))
    return idx_zero

def get_neighbors(grid_size, boundaries):
    neighbors = np.zeros((grid_size, 4), dtype=int)

    for i in range(grid_size):
        neighbors[i] = idx_neighbor_sides(i)

    return neighbors

boundaries = idx_boundary_sides()
neighbors = get_neighbors(N, boundaries)

idx_zero = np.array(idx_zero_sides(N, boundaries))
size_zero = len(idx_zero)

def get_hamilton(J, Bz, K, Dx, Dy, nset, neigh):
    e = 0

    e += -J*np.sum(np.einsum('ac, abc -> a', nset, neigh[:,[0,2]], optimize=True))

    e += -Dx*np.sum(np.cross(nset, neigh[:,2])[:,0])
    e += +Dy*np.sum(np.cross(nset, neigh[:,0])[:,1])

    e += -K*np.dot(nset[:,2], nset[:,2])

    e += -Bz*np.sum(nset[:,2])

    return e

def get_derivative_exchange(J, neigh):
    return -J*np.einsum('abc -> ac', neigh, optimize=True)

dx_array = -Dx*np.array([1,-1])
dy_array = -Dy*np.array([1,-1])
def get_derivative_dm_x(neigh):
    return np.einsum('ab, b -> a', neigh, dx_array, optimize=True)

def get_derivative_dm_y(neigh):
    return np.einsum('ab, b -> a', neigh, dy_array, optimize=True)

def get_derivative_dm_z(neigh):
    return -np.einsum('ab, b -> a', neigh[:,[0,1]][:,:,0], dy_array, optimize=True)-np.einsum('ab, b -> a', neigh[:,[2,3]][:,:,1], dx_array, optimize=True)

def get_Beff(mu, nset):
    Beff = np.zeros((N, 3))

    nset_neigh = nset[neighbors]
    nset_neigh[idx_zero[:, 0], idx_zero[:, 1]] = np.zeros((size_zero, 3))


    Beff += get_derivative_exchange(J, nset_neigh)

    Beff[:,0] += get_derivative_dm_y(nset_neigh[:,[0,1]][:,:,2])
    Beff[:,1] += get_derivative_dm_x(nset_neigh[:,[2,3]][:,:,2])
    Beff[:,2] += get_derivative_dm_z(nset_neigh) -Bz*np.ones(N) - 2*K*nset[:,2]

    Beff *= -(1/mu)
    # Set or remove boundary cond

    #Beff[10:Nx] = np.zeros((Nx-10, 3))
    #Beff[-Nx+10:] = np.zeros((Nx-10, 3))
    #for i in range(Ny):
    #    Beff[Nx*i] = np.zeros(3)
    #    Beff[Nx*i+Nx-1] = np.zeros(3)

    return Beff

def get_periods(nrow, row_length):
    nrow_init = nrow[0]-np.dot(nrow[0][1],ey)
    rotations = 0
    orthogonal = np.cross(ey, nrow_init)
    for i in range(1, row_length-1):
        #if np.sign(np.dot(nrow_init,nrow[i]-np.dot(nrow[i],ey))) == -np.sign(np.dot(nrow[i+1]-np.dot(nrow[i+1],ey), nrow_init)):
        if np.sign(np.dot(orthogonal, nrow[i])) > np.sign(np.dot(orthogonal, nrow[i+1])):
            rotations += 1

    partial_angle = np.arccos(np.dot(nrow_init, nrow[-1]))

    if np.dot(orthogonal, nrow[-1]) < 0:
        rotations += partial_angle/(2*pi)
    else:
        rotations += 1 - partial_angle/(2*pi)

    return rotations

def rotate_edge(nedge, v, t, dt):
    rotmat = np.array([[cos(v*t), 0, sin(v*t)], [0, 1, 0], [-sin(v*t), 0, cos(v*t)]])
    nedge_new = np.copy(nedge)
    for i in range(len(nedge)):
        nedge_new[i] = np.dot(rotmat, nedge[i])
    return nedge_new
    #return np.einsum('bc, ac -> ab', rotmat, nedge)

def apply_edge_field(edge_length, B0, v, t):
    B = -B0*np.array([-sin(v*t), 0, -cos(v*t)])
    return np.tensordot(np.ones(edge_length), B, axes=0)

def apply_edge_field_switch(edge_length, mu, B0, steps, period, tswitch, t_init):
    ones = np.ones(edge_length)
    B = -B0*np.array([-sin(-np.pi/4*0.8), 0, -cos(-np.pi/4*0.8)])
    del_period = period-tswitch
    nsteps = steps+del_period-t_init
    if nsteps%period <= del_period:
        return np.tensordot(np.ones(edge_length), B, axes=0)
    else:
        return -np.tensordot(np.ones(edge_length), B, axes=0)

def switch_magnetization(nedge, steps, period, tswitch, t_init):
    del_period = period-tswitch
    nsteps = steps+del_period-t_init
    if nsteps%period <= del_period:
        return nedge
    else:
        return -nedge

x_grid = xcart.flatten()
y_grid = ycart.flatten()
def get_skyr_pos(nset):
    min_pos = np.argmin(nset[:,2], axis=0)
    args = np.array([-Nx,-1,1,Nx])
    min_square = min_pos + args

    n_sqr = nset[min_square]
    n_min = nset[min_pos]

    xmin = 0
    ymin = 0

    if np.sign(n_sqr[2][0])-np.sign(n_min[0]) != 0:
        xmin = -n_min[0]/(n_sqr[2][0] - n_min[0])
    else:
        xmin = n_min[0]/(n_sqr[1][0] - n_min[0])

    if np.sign(n_sqr[3][1])-np.sign(n_min[1]) != 0:
        ymin = -n_min[1]/(n_sqr[3][1] - n_min[1])
    else:
        ymin = n_min[1]/(n_sqr[0][1] - n_min[1])

    return np.array([x_grid[min_pos] + xmin, y_grid[min_pos] + ymin])


def get_topological_charge(nset, neigh):
    s23 = np.einsum('ab, ab -> a', nset, neigh[:,0])
    s12 = np.einsum('ab, ab -> a', nset, neigh[:,2])
    s31 = np.einsum('ab, ab -> a', neigh[:,0], neigh[:,2])
    scross = np.einsum('ab, ab -> a', neigh[:,2], np.cross(nset, neigh[:,0]))

    s24 = np.einsum('ab, ab -> a', nset, neigh[:,1])
    s45 = np.einsum('ab, ab -> a', neigh[:,1], neigh[:,3])
    s52 = np.einsum('ab, ab -> a', nset, neigh[:,3])
    scross2 = np.einsum('ab, ab -> a', nset, np.cross(neigh[:,1], neigh[:,3]))

    p1 = np.sqrt(2*(1+s12)*(1+s23)*(1+s31))
    exp_omega1 = (1+s12+s23+s31+1j*scross)/p1
    omega1 = 2*np.imag(np.log(exp_omega1))

    p2 = np.sqrt(2*(1+s24)*(1+s45)*(1+s52))
    exp_omega2 = (1+s24+s45+s52+1j*scross2)/p2
    omega2 = 2*np.imag(np.log(exp_omega2))

    return np.sum(omega1+omega2)/(4*pi)

n0 = np.copy(n_init)
idx_u = np.delete(np.arange(0, N, 1, dtype=int)[Nx:-Nx], slice(None, None, Nx), 0)
idx_nset_inner = np.delete(idx_u, slice(Nx-2, None, Nx-1), 0)

gamma = 1
#gamma_e = 1.76
# 1.76*10**11 rad Hz T^-1
alpha = 0.1
n_t = np.copy(n0)

def get_derivative(n_t, Beff):
    n_x_Beff = np.cross(n_t, Beff)
    return -gamma*(n_x_Beff + alpha*np.cross(n_t, n_x_Beff))/(1+alpha**2)

Bset = np.arange(1, 31, 1, dtype=int)
B_length = len(Bset)
edge_set = []
idx_edges = np.arange(0+7*Nx, N-7*Nx, Nx, dtype=int)
edge_set.append(idx_edges)
for i in range(1, 10):
    idx_edges = np.concatenate((idx_edges, np.arange(i+7*Nx, i+N-7*Nx, Nx, dtype=int)))
    edge_set.append(idx_edges)

# Solve Euler-Equation
dt = 0.00176
t = 0
v = 1*0.357

for i in range(10000):
    Beff = get_Beff(mu, n_t)
    K1 = get_derivative(n_t, Beff)

    Beff = get_Beff(mu, n_t + dt*K1)
    K2 = get_derivative(n_t + dt*K1, Beff)

    n_t = n_t + (dt/2)*(K1+K2)

    n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
    t += dt

n_t_init = np.copy(n_t)

results = np.zeros((len(edge_set), B_length), dtype=int)

def simulate(result, index):
    #p = multiprocessing.current_process()
    #index = p._identity[0] - 1

    idx_edge = edge_set[index]
    edge_length = len(idx_edge)
    bool_array = np.zeros(B_length, dtype=int)

    for j in range(B_length):
        Bj = Bset[j]
        n_t = np.copy(n_t_init)
        t = 0

        for _ in range(10000):
            Beff = get_Beff(mu, n_t)
            Beff[idx_edge] += apply_edge_field(edge_length, Bj, v, t)
            K1 = get_derivative(n_t, Beff)

            Beff = get_Beff(mu, n_t + dt*K1)
            Beff[idx_edge] += apply_edge_field(edge_length, Bj, v, t + dt)
            K2 = get_derivative(n_t + dt*K1, Beff)

            n_t = n_t + (dt/2)*(K1+K2)

            n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
            t += dt

        for _ in range(10000):
            Beff = get_Beff(mu, n_t)
            K1 = get_derivative(n_t, Beff)

            Beff = get_Beff(mu, n_t + dt*K1)
            K2 = get_derivative(n_t + dt*K1, Beff)

            n_t = n_t + (dt/2)*(K1+K2)

            n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
            t += dt

        if np.abs(get_topological_charge(n_t, n_t[neighbors])) >= 0.9 and np.min(n_t[boundaries][:,2]) > 0:
        #if np.abs(get_topological_charge(n_t, n_t[neighbors])) >= 0.9 and n_t[:Ny]:
            bool_array[j] = 1
        #print("Bvalue " + str(j) + " done")
    print(bool_array)
    result[index] = list(bool_array)


threads = [None] * len(edge_set)
manager = multiprocessing.Manager()
lst = manager.list()
for i in range(len(edge_set)):
    lst.append([0] * B_length)

for i in range(len(threads)):
    threads[i] = multiprocessing.Process(target=simulate, args=(lst, i))
    threads[i].start()

for i in range(len(threads)):
    threads[i].join()

plt.figure(figsize=(12,4))
for i in range(len(edge_set)):
    plt.scatter(Bset, np.repeat(i+1, B_length), c=lst[i], cmap='RdYlGn', s=400, marker='s')
plt.xlabel("B [T]", fontsize=12)
plt.ylabel("Number of edge rows", fontsize=12)
plt.show()

print(lst)
