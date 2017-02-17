import numpy as np
#import scipy.optimize as optimize

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

#################################### Grids #####################################

def uniform_random_samples(npoints, limits):
    """Select parameters in a hypercube sampled uniformly in the range limits.
    i.e. Monte Carlo sampling.

    Parameters
    ----------
    npoints : int
        Number of points to select.
    limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
        Boundaries of the hypercube.
    """
    ndim = len(limits)

    param_columns = []
    for i in range(ndim):
        xi_min, xi_max = limits[i]
        xis = np.random.uniform(low=xi_min, high=xi_max, size=npoints)
        param_columns.append(xis)

    return np.array(param_columns).T


def latin_hypercube(npoints, limits):
    """Select parameters using the Latin Hypercube method.

    Parameters
    ----------
    npoints : int
        Number of points to select.
    limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
        Boundaries of the hypercube.
    """
    ndim = len(limits)

    # Make a list of points
    # Make sure they are floats and not ints
    points = np.array([[1.0*i]*ndim for i in range(npoints)])

    # Shuffle the values for each dimension
    for j in range(ndim):
        np.random.shuffle(points[:, j])

    # Rescale the points to match the bounds from limits
    for j in range(ndim):
        low = float(limits[j, 0])
        high = float(limits[j, 1])
        points[:, j] = low + points[:, j]*(high-low)/(npoints-1.0)

    return points


# def latin_hypercube(Ndata, limits):
#     """Select parameters using the Latin Hypercube method.
#
#     Parameters
#     ----------
#     Ndata : int
#         Number of points to select.
#     limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
#         Boundaries of the hypercube.
#     """
#     Nparams = len(limits)
#
#     # Find indices of grid points
#     point_ind_list = []
#     allowed_indices = np.array([[n for n in range(Ndata)] for i in range(Nparams)])
#     for n in range(Ndata):
#         js = np.random.randint(Ndata-n, size=Nparams)
#         point_ind = [allowed_indices[i, js[i]] for i in range(Nparams)]
#         point_ind_list.append(point_ind)
#         allowed_indices = np.array([np.delete(allowed_indices[i], js[i]) for i in range(Nparams)])
#
#     point_ind_array = np.array(point_ind_list)
#
#     # Get grid points
#     grids = []
#     for i in range(Nparams):
#         grids.append(np.linspace(limits[i, 0], limits[i, 1], Ndata))
#
#     grids_array = np.array(grids)
#
#     # Get chosen points on the grid
#     points = np.array([[grids_array[i, point_ind_array[n, i]] for i in range(Nparams)] for n in range(Ndata)])
#
#     return points
