import mdtraj
import numpy as np
import scipy.spatial as ss
import subprocess
from joblib import Parallel, delayed, cpu_count


def get_neighbor_distances(n_neighbors, reference_structure, topfile):

    reference_structure = mdtraj.load_lammpstrj(reference_structure, top=topfile)
    natoms = reference_structure.n_atoms
    coords = 100.0*reference_structure.xyz*reference_structure.unitcell_lengths

    center = np.mean(coords, axis=1)
    center_ind = np.argmin(np.sqrt(np.sum((coords - center)**2.0, axis=2)))

    tree = ss.cKDTree(coords[0, :, :])
    (neighbor_distances, ii) = tree.query(coords[0, center_ind, :], n_neighbors)

    return neighbor_distances


def compute_psi(box_sizes, coords, neighbor_distances, smoothing_cutoff):

    n_neighbors = len(neighbor_distances)

    tree = ss.cKDTree(coords[0, :, :], boxsize=np.hstack((box_sizes, box_sizes)))
    (dd, ii) = tree.query(coords[0, :, :], n_neighbors+1)
    phi = np.sum((dd[:, 1:] - neighbor_distances)**2, axis=1)/n_neighbors
    del dd, ii

    tree = ss.cKDTree(np.transpose(coords[0, :, [0, 2]]),
                      boxsize=np.hstack((box_sizes[0, [0, 2]], box_sizes[0, [0, 2]])))

    try:

        (dist, ii) = tree.query(np.transpose(coords[0, :, [0, 2]]),
                                compute_psi.n_neighbors)

    except TypeError:

        neighbors = tree.query_ball_point(np.transpose(coords[0, :, [0, 2]]),
                                          smoothing_cutoff)

        compute_psi.n_neighbors = int(1.05*np.max([len(n) for n in neighbors]))

        (dist, ii) = tree.query(np.transpose(coords[0, :, [0, 2]]),
                                compute_psi.n_neighbors)

    wd = ((1.0 - (dist/smoothing_cutoff)**2)**2)*dist* \
         (dist < smoothing_cutoff).astype(int)
    del dist

    psi = np.sum(wd*phi[ii], axis=1)/np.sum(wd, axis=1)
    del wd, ii

    return (phi, psi)


if __name__ == "__main__":

    latparam = 2.92625

    neighbor_distances = get_neighbor_distances(14, 'traj_Fe/reference_solid_Fe.lammpstrj',
                                                'traj_Fe/reference_solid_Fe.pdb')

    traj = mdtraj.load_lammpstrj('traj_Fe/Fe.lammpstrj', top='traj_Fe/Fe.pdb')
    nframes = traj.n_frames

    compute_psi.n_neighbors = None
    phi_psi1 = compute_psi(10.0*traj[0].unitcell_lengths, 10.0*traj[0].xyz,
                           neighbor_distances, 2.5*latparam)

    phi_psi2 = np.array(Parallel(n_jobs=10) \
                       (delayed(compute_psi) \
                        (10.0*traj[frame].unitcell_lengths, 10.0*traj[frame].xyz,
                         neighbor_distances, 2.5*latparam) \
                        for frame in range(1, nframes)))
