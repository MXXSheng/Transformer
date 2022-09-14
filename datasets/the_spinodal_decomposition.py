import os
import time
import numpy as np
import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

from joblib import Parallel, delayed

def write_vtk_grid_values(nx, ny, dx, dy, step, data1, time_energy_out_path):
    path = os.path.join(time_energy_out_path, "vtk")
    # print(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file = open(os.path.join(path, "time_{:0>5}.vtk".format(step)), 'w')
    nz = 1
    npoint = nx * ny * nz

    file.write("# vtk DataFile Version 2.0\n"
               "time_10.vtk\n"
               "ASCII\n"
               "DATASET STRUCTURED_GRID\n"
               "DIMENSIONS {:5d} {:5d} {:5d}\n"
               "POINTS {:7d} float\n".format(nx, ny, nz, npoint)
               )
    for i in range(nx):
        for j in range(ny):
            x = i * dx
            y = j * dy
            z = 0.0
            file.write("{:.6e} {:.6e} {:.6e}\n".format(x, y, z))

    file.write("POINT_DATA {:5d}\n"
               "SCALARS CON float 1\n"
               "LOOKUP_TABLE default\n".format(npoint))

    for i in range(nx):
        for j in range(ny):
            file.write("{:.6e} ".format(data1[i, j]))
    file.close()
def write_time_matrix_values(step, data, time_energy_out_path):
    path = os.path.join(time_energy_out_path, "matrices")
    if not os.path.isdir(path):
        os.makedirs(path)
    # print(data.shape)
    # plt.plot(data)
    # plt.savefig("images/time{:0>5}.jpg".format(step))

    # matplotlib.image.imsave("images/time{:0>5}.png".format(step), data)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, "time{:0>5}.csv".format(step)), index=False, header=False)
def toeplitz(c, r):  # 生成托普利茨矩阵
    c = np.array(c)
    r = np.array(r)

    # 将c与r的全部元素构成列向量
    m, n = c.shape
    y, z = r.shape
    temp1 = []
    temp2 = []
    for i in range(n):
        for temp in c:
            temp1.append(temp[i])
    for i in range(z):
        for temp in r:
            temp2.append(temp[i])

    c = temp1
    r = temp2

    p = len(r)
    m = len(c)

    x = list(r[p - 1:0:-1])
    for i in c:
        x.append(i)

    temp3 = np.arange(0, m)
    temp4 = np.arange(p - 1, -1, -1)

    temp3.shape = (m, 1)
    temp4.shape = (1, p)

    ij = temp3 + temp4
    t = np.array(x)[ij]

    return t
def laplacian(nx, ny, dx, dy):
    """

    :param nx:
    :param ny:
    :param dx:
    :param dy:
    :return:
    This function generates the Laplace operator by using five-point stencil with periodic boundaries.
    """
    nxny = nx * ny
    r = np.zeros([1, nx])
    r[0, 0] = 2
    r[0, 1] = -1
    T = toeplitz(r, r)
    E = np.eye(nx)
    grad = -(np.kron(T, E) + np.kron(E, T))
    for i in range(nx):
        ii = i * nx
        jj = ii + nx - 1
        grad[ii, jj] = 1.0
        grad[jj, ii] = 1.0

        kk = nxny - nx + i
        grad[i, kk] = 1.0
        grad[kk, i] = 1.0

    grad = grad / (dx * dy)
    return grad
def micro_ch_pre(nx, ny, c0):
    """

    :param nx:
    :param ny:
    :param c0:
    :return: con

    This function initializes the microstructure for given average
    composition modulated with a noise term to account the thermal
    fluctuations in Cahn–Hilliard equation.
    """
    nxny = nx * ny
    noise = 0.02
    con = np.zeros([nxny, 1])

    for i in range(nx):
        for j in range(ny):
            con[i * nx + j] = c0 + noise * (0.5 - random.random())
    print("Init: con = ", con)
    return con
def free_energ_ch_v2(nx, ny, con):
    """

    :param nx:
    :param ny:
    :param con:
    :return:
    This function calculates the derivative of free energy simultaneously at all the grid points in the simulation cell.
    """
    A = 1.0
    dfdcon = A * (2.0 * con * (1 - con)**2 - 2.0 * con**2 * (1 - con))
    return dfdcon
def calculate_enery(nx, ny, con, grad_coef):
    """

    :param nx:
    :param ny:
    :param con:
    :param grad_coef:
    :return:
    This function calculates the total bulk energy of the system.
    """
    energy = 0.0
    for i in range(nx-1):
        for j in range(ny-1):
            energy = energy + con[i, j]**2 * (1-con[i, j])**2 + 0.5 * grad_coef * ((con[i+1, j] - con[i, j])**2 + (con[i, j+1] - con[i, j])**2)
    return energy



def main():
    time0 = time.time()
    # print(time0)  # 开始时间
    nx = 64
    ny = 64
    nxny = nx * ny
    dx = 1.0
    dy = 1.0
    nstep = 3000
    nprint = 100
    dtime = 1.0e-2
    ttime = 0
    c0 = 0.40
    # mobility = 1.0
    grad_coef = 0.5

    if not os.path.isdir("data"):
        os.makedirs("data")
    path = os.path.join(os.getcwd(), "data")

    Parallel(n_jobs=20)(
        delayed(function)(path, mobility, nx, ny, time0, nxny, dx, dy, nstep, nprint, dtime, ttime, c0, grad_coef)
        for mobility in np.arange(0.5, 1.5, 0.001))

    # for mobility in np.arange(0.5, 1.5, 0.001):
    #     function(path=path, mobility=mobility, nx=nx, ny=ny, time0=time0, nxny=nxny, dx=dx, dy=dy, nstep=nstep,
    #              nprint=nprint, dtime=dtime, ttime=ttime, c0=c0, grad_coef=grad_coef)


def function(path, mobility, nx, ny, time0, nxny, dx, dy, nstep, nprint, dtime, ttime, c0, grad_coef):
    # 创建mobility文件夹
    if not os.path.isdir(os.path.join(path, "mobility={:.3f}".format(mobility))):
        os.makedirs(os.path.join(path, "mobility={:.3f}".format(mobility)))
    # 创建time_energy_out文件
    time_energy_out_path = os.path.join(path, "mobility={:.3f}".format(mobility))
    file_out = open(os.path.join(time_energy_out_path, "time_energy_out"), "w")

    con = micro_ch_pre(nx, ny, c0)

    grad = laplacian(nx, ny, dx, dy)
    # print("grad = ", grad)
    for istep in range(nstep):
        ttime += dtime

        dfdcon = free_energ_ch_v2(nx, ny, con)
        lap_con2 = grad @ (dfdcon - grad_coef * grad @ con)
        # print(grad.shape)
        # print(con.shape)
        # print(np.dot(grad, con).shape)
        con = con + dtime * mobility * lap_con2
        # print(con)
        con[con > 0.9999] = 0.9999
        con[con < 0.00001] = 0.00001
        if istep % nprint == 0 or istep == 1:
            file_out.write("Done step: {:5d}\n".format(istep))

        con2 = np.zeros([nx, ny])
        for i in range(nx):
            for j in range(ny):
                ii = i * nx + j
                # print(con)
                con2[i, j] = con[ii]

        #  -----------calculate total energy-----------
        energy = calculate_enery(nx, ny, con2, grad_coef)
        file_out.write("{} {}".format(ttime, energy))
        if istep % 3 == 0:
            write_vtk_grid_values(nx, ny, dx, dy, istep, con2, time_energy_out_path)
            write_time_matrix_values(istep, con2, time_energy_out_path)
        #  ------------calculate compute time----------------
        print("step{} Compute Time:{}".format(istep, time.time() - time0))


if __name__ == '__main__':
    main()

# def write_vtk_grid_values(nx, ny, dx, dy, step, data1):
#     path = os.path.abspath(os.path.join(os.getcwd(), "vtk\\time_{}.vtk".format(istep)))
#     print(path)
#     file = open(path, 'w')
#
#     pass
