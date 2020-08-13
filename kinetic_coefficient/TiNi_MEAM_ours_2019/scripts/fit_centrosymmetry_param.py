import numpy as np

def erf_fit_func(order_param_sol, order_param_liq, interface_flag,
                 pos, pos_interface, sigma):

    import scipy.special as ss

    order_param_fit = 0.5*((order_param_sol + order_param_liq) + interface_flag* \
             (order_param_sol - order_param_liq)*ss.erf((pos - pos_interface)/ \
                                                        (sigma*np.sqrt(2.0))))

    return order_param_fit


def residual(params, interface_flag, pos_top, pos_bottom, pos, order_param, wghts):
    """
    Description
    ----
    Compute residuals for fit.

    Inputs
    ----
    :params: Parameters for the model
    :interface_flag: 1 for upper interface and -1 for lower interface
    :pos_top: Position of top layer of solid
    :pos_bottom: Position of bottom layer of solid
    :pos: Position in interface normal direction
    :order_param: Order parameter to distinguish liquid from solid
    :wghts: Weights

    Outputs
    ----
    :residuals: Residuals
    """

    pos_center = 0.5*(pos_top - pos_bottom) + pos_bottom
    pos_upper = 0.95*(pos_top - pos_bottom) + pos_bottom
    pos_lower = 0.05*(pos_top - pos_bottom) + pos_bottom

    order_param_sol = params['order_param_sol'].value
    order_param_liq = params['order_param_liq'].value
    sigma = params['sigma'].value
    pos_interface = params['pos_interface'].value

    if interface_flag == 1:
        ind = np.intersect1d(np.where(pos > pos_center)[0], np.where(pos < pos_upper)[0])
    elif interface_flag == -1:
        ind = np.intersect1d(np.where(pos > pos_lower)[0], np.where(pos < pos_center)[0])

    pos = pos[ind]
    order_param = order_param[ind]
    wghts = wghts[ind]

    model = erf_fit_func(order_param_sol, order_param_liq, interface_flag,
                         pos, pos_interface, sigma)

    residuals = (order_param - model)*wghts

    return residuals


def fitting(pos_bottom, pos_top, pos, order_param):

    from lmfit import minimize, Parameters

    wghts = np.ones(len(pos))  # All weights equal
    params = Parameters()

    try:
        params.add('order_param_sol', value=fitting.order_param_sol_ini, min=0.0)
    except AttributeError:
        pos_lower = 0.05*(pos_top - pos_bottom) + pos_bottom
        pos_upper = 0.1*(pos_top - pos_bottom) + pos_bottom
        ind1 = np.intersect1d(np.where(pos > pos_lower)[0], np.where(pos < pos_upper)[0])
        pos_lower = 0.90*(pos_top - pos_bottom) + pos_bottom
        pos_upper = 0.95*(pos_top - pos_bottom) + pos_bottom
        ind2 = np.intersect1d(np.where(pos > pos_lower)[0], np.where(pos < pos_upper)[0])
        ind = np.union1d(ind1, ind2)
        fitting.order_param_sol_ini = np.mean(order_param[ind])
        params.add('order_param_sol', value=fitting.order_param_sol_ini, min=0.0)

    try:
        params.add('order_param_liq', value=fitting.order_param_liq_ini, min=0.0)
    except AttributeError:
        pos_lower = 0.45*(pos_top - pos_bottom) + pos_bottom
        pos_upper = 0.55*(pos_top - pos_bottom) + pos_bottom
        ind = np.intersect1d(np.where(pos > pos_lower)[0],
                             np.where(pos < pos_upper)[0])
        fitting.order_param_liq_ini = np.mean(order_param[ind])
        params.add('order_param_liq', value=fitting.order_param_liq_ini, min=0.0)

    try:
        params.add('pos_interface', value=fitting.pos_interface_lower_ini)
    except AttributeError:
        fitting.pos_interface_lower_ini = 0.125*(pos_top - pos_bottom) + pos_bottom
        params.add('pos_interface', value=fitting.pos_interface_lower_ini)

    try:
        params.add('sigma', value=fitting.sigma_ini, min=0.0)
    except AttributeError:
        params.add('sigma', value=3.0, min=0.0)

    fit = minimize(residual, params,
                   args=(-1, pos_top, pos_bottom, pos, order_param, wghts))

    order_param_liq_lower = fit.params['order_param_liq'].value
    order_param_sol_lower = fit.params['order_param_sol'].value
    pos_interface_lower = fit.params['pos_interface'].value
    sigma_lower = fit.params['sigma'].value

    params.pop('pos_interface')
    try:
        params.add('pos_interface', value=fitting.pos_interface_upper_ini)
    except AttributeError:
        pos_center = 0.5*(pos_top - pos_bottom) + pos_bottom
        fitting.pos_interface_upper_ini = 2.0*pos_center - pos_interface_lower
        params.add('pos_interface', value=fitting.pos_interface_upper_ini)


    fit = minimize(residual, params,
                   args=(1, pos_top, pos_bottom, pos, order_param, wghts))

    order_param_liq_upper = fit.params['order_param_liq'].value
    order_param_sol_upper = fit.params['order_param_sol'].value
    pos_interface_upper = fit.params['pos_interface'].value
    sigma_upper = fit.params['sigma'].value

    fitting.order_param_sol_ini = (order_param_sol_lower + order_param_sol_upper)/2.0
    fitting.order_param_liq_ini = (order_param_liq_lower + order_param_liq_upper)/2.0
    fitting.sigma_ini = (sigma_lower + sigma_upper)/2.0
    fitting.pos_interface_lower_ini = pos_interface_lower
    fitting.pos_interface_upper_ini = pos_interface_upper

    return (order_param_sol_lower, order_param_liq_lower, sigma_lower,
            pos_interface_lower, order_param_sol_upper, order_param_liq_upper,
            sigma_upper, pos_interface_upper)


def getdatafit(lmpptr, time, outfile):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    from lammps import lammps
    lmp = lammps(ptr=lmpptr)

    natoms = lmp.get_natoms()

    if rank == 0:
        order_param = np.empty([size, natoms], dtype='float')
        pos = np.empty([size, natoms], dtype='float')
    else:
        order_param = np.empty(natoms, dtype='float')
        pos = np.empty(natoms, dtype='float')

    p = np.array(lmp.extract_variable("pos", "all", 1))
    p = np.hstack((p, np.nan*np.ones(natoms-len(p))))

    o = np.array(lmp.extract_variable("centro", "all", 1))
    o = np.hstack((o, np.nan*np.ones(natoms-len(o))))

    comm.Gather(o, order_param, root=0)
    comm.Gather(p, pos, root=0)

    if rank == 0:
        order_param = order_param[~np.isnan(order_param)]
        pos = pos[~np.isnan(pos)]

    pos_top = lmp.extract_variable("postop", "all", 0)
    pos_bottom = lmp.extract_variable("posbottom", "all", 0)

    params = np.empty(4)

    if rank == 0:
        (order_param_sol_lower, order_param_liq_lower, sigma_lower, pos_interface_lower,
        order_param_sol_upper, order_param_liq_upper, sigma_upper,
        pos_interface_upper) = fitting(pos_bottom, pos_top, pos, order_param)
        
        params = np.array([(order_param_sol_lower + order_param_sol_upper)/2.0,
                  (order_param_liq_lower + order_param_liq_upper)/2.0,
                  (sigma_lower + sigma_upper)/2.0,
                  pos_interface_upper - pos_interface_lower])

        with open(outfile, 'a') as f:
            outdata = str(time) + ' ' + ' '.join(params.astype(str)) + '\n'
            f.write(outdata)

    comm.Bcast(params, root=0)
    
    return params[3]


# def readtrajectoryfit(traj_file, outfile, plot_flag=False):
# 
    # import subprocess
# 
    # try:
        # nskip = int(subprocess.Popen('cat ' + traj_file + \
                                     # ' | awk \'$1 == "ITEM:" {print NR}\' | tail -1',
                                     # shell=True,
                                     # stdout=subprocess.PIPE).stdout.readline().rstrip('\n'))
    # except ValueError:
        # nskip = 0
# 
    # data = np.loadtxt(traj_file, skiprows=nskip)
    # atom_type = data[:, 1]
    # pos = data[:, 4]
    # order_param = data[:, -1]
    # del data
# 
    # ind = np.where(atom_type == 3)[0]
    # pos_bottom = np.mean(pos[ind])
# 
    # ind = np.where(atom_type == 5)[0]
    # pos_top = np.mean(pos[ind])
# 
    # (order_param_sol_lower, order_param_liq_lower, sigma_lower, pos_interface_lower,
     # order_param_sol_upper, order_param_liq_upper, sigma_upper, pos_interface_upper) = \
    # fitting(pos_bottom, pos_top, pos, order_param)
# 
    # params = np.array([(order_param_sol_lower + order_param_sol_upper)/2.0,
                       # (order_param_liq_lower + order_param_liq_upper)/2.0,
                       # (sigma_lower + sigma_upper)/2.0,
                       # pos_interface_upper - pos_interface_lower], dtype=str)
# 
    # with open(outfile, 'a') as f:
        # f.write(' '.join(params) + '\n')
# 
    # if plot_flag:
        # return (order_param_sol_lower, order_param_liq_lower, sigma_lower,
                # pos_interface_lower, order_param_sol_upper, order_param_liq_upper,
                # sigma_upper, pos_interface_upper)
    # else:
        # return pos_interface_upper - pos_interface_lower
# 
# 
# def plot_fits():
# 
    # (order_param_sol_lower, order_param_liq_lower, sigma_lower,
            # pos_interface_lower, order_param_sol_upper, order_param_liq_upper,
            # sigma_upper, pos_interface_upper) = read_trajectory_fit('frame0.dat')
# 
    # import matplotlib.pyplot as plt
    # import my_plot_settings_article as mpsa
# 
    # plt.plot(pos, order_param, '.', markersize=0.5, label='data')
    # plt.ylim(0, 17)
# 
    # pos_center = 0.5*(pos_top - pos_bottom) + pos_bottom
# 
    # ind = np.where(pos < pos_center)[0]
    # pos_plt = np.sort(pos[ind])
    # plt.plot(pos_plt, erf_fit_func(order_param_sol_lower, order_param_liq_lower, -1,
                                   # pos_plt, pos_interface_lower, sigma_lower),
             # 'g', linewidth=1.5, label='lower fit')
    # plt.plot(pos_interface_lower*np.ones(2), [plt.ylim()[0], 12.0], 'g--', linewidth=1.5,
             # label='lower interface')
# 
    # ind = np.where(pos > pos_center)[0]
    # pos_plt = np.sort(pos[ind])
    # plt.plot(pos_plt, erf_fit_func(order_param_sol_upper, order_param_liq_upper, 1,
                                   # pos_plt, pos_interface_upper, sigma_upper),
             # 'r', linewidth=1.5, label='upper fit')
    # plt.plot(pos_interface_upper*np.ones(2), [plt.ylim()[0], 12.0], 'r--', linewidth=1.5,
             # label='upper interface')
# 
    # mpsa.axis_setup('x')
# 
    # plt.xlabel('$\mathrm{z/z_{box}}$', labelpad=mpsa.axeslabelpad)
    # plt.ylabel('centrosymmetry parameter', labelpad=mpsa.axeslabelpad)
    # plt.legend(ncol=2)
# 
    # mpsa.save_figure('centrosymmetry_fits.png')
    # plt.close()


if __name__ == "__main__":
    
    try: del fitting.order_param_sol_ini
    except AttributeError: pass
    
    try: del fitting.order_param_liq_ini
    except AttributeError: pass
    
    try: del fitting.sigma_ini
    except AttributeError: pass
    
    try: del fitting.pos_interface_lower_ini
    except AttributeError: pass
    
    try: del fitting.pos_interface_upper_ini
    except AttributeError: pass
    
    # dist = readtrajectoryfit('../output/1690.0K_boundary_xy_springs_all_langevin/frame0.dat',
                             # 'test.dat')
    # print(dist)
