import numpy as np
import scipy.optimize as optimize
import copy
import h5py

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

import designofexperiment as doe

###################### Wrappers for scikit-learn functions #####################

def generate_gp(points, data, hp0, kernel_type='squaredexponential',
                fixed=False, hyper_limits=None, n_restarts_optimizer=9):
    """Gaussian Process for ndim dimensional parameter space.

    Parameters
    ----------
    points : array of shape (npoints, ndim).
        Coordinates in paramete space of sampled data.
    data : array of shape (npoints,).
        Data at each of the sampled points.
    hp0 : array of shape (ndim+2,)
        Initial hyperparameter guess for optimizer.
        Order is (sigma_f, ls_0, ls_1, ..., sigma_n).
    kernel_type : 'squaredexponential', 'matern32', 'matern52'
    limits : array of shape (ndim+2, 2)
        Lower and upper bounds on the value of each hyperparameter.
    n_restarts_optimizer : int
        Number of random points in the hyperparameter space to restart optimization
        routine for searching for the maximum log-likelihood.
        Total number of optimizations will be n_restarts_optimizer+1.

    Returns
    -------
    gp : GaussianProcessRegressor
    """

    # ******* Generate kernel *******

    # ConstantKernel = c multiplies *all* elements of kernel matrix by c
    # If you want to specify sigma_f (where c=sigma_f^2) then use
    # sigma_f^2 and bounds (sigma_flow^2, sigma_fhigh^2)

    # WhiteKernel = c \delta_{ij} multiplies *diagonal* elements by c
    # If you want to specify sigma_n (where c=sigma_n^2) then use
    # sigma_n^2 and bounds (sigma_nlow^2, sigma_nhigh^2)

    # radial part uses the length scales [l_0, l_1, ...] not [l_0^2, l_1^2, ...]

    # Constant and noise term
    if fixed==True:
        const = ConstantKernel(hp0[0]**2)
        noise = WhiteKernel(hp0[-1]**2)
    elif fixed==False:
        const = ConstantKernel(hp0[0]**2, hyper_limits[0]**2)
        noise = WhiteKernel(hp0[-1]**2, hyper_limits[-1]**2)
    else:
        raise Exception, "'fixed' must be True or False."

    # Radial term
    if fixed==True:
        if kernel_type=='squaredexponential':
            radial = RBF(hp0[1:-1])
        elif kernel_type=='matern32':
            radial = Matern(hp0[1:-1], nu=1.5)
        elif kernel_type=='matern52':
            radial = Matern(hp0[1:-1], nu=2.5)
        else: raise Exception, "Options for kernel_type are: 'squaredexponential', 'matern32', 'matern52'."
    elif fixed==False:
        if kernel_type=='squaredexponential':
            radial = RBF(hp0[1:-1], hyper_limits[1:-1])
        elif kernel_type=='matern32':
            radial = Matern(hp0[1:-1], hyper_limits[1:-1], nu=1.5)
        elif kernel_type=='matern52':
            radial = Matern(hp0[1:-1], hyper_limits[1:-1], nu=2.5)
        else: raise Exception, "Options for kernel_type are: 'squaredexponential', 'matern32', 'matern52'."
    else:
        raise Exception, "'fixed' must be True or False."

    kernel = const * radial + noise

    # ******* Initialize GaussianProcessRegressor and optimize hyperparameters if not fixed *******

    if fixed==True:
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        # Supply the points and data, but don't optimize the hyperparameters
        gp.fit(points, data)
        return gp
    elif fixed==False:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        # Optimize the hyperparameters by maximizing the log-likelihood
        gp.fit(points, data)
        return gp
    else:
        raise Exception, "'fixed' must be True or False."


def get_hyperparameters(gp):
    """Get the hyperparameters of the GaussianProcessRegressor gp.
    The order is (sigma_f, ls_0, ls_1, ..., sigma_n).
    """
    #print gp.kernel
    #print gp.kernel_
    #print gp.kernel(points[:5])

    # kernel is what is used to initialize GaussianProcessRegressor
    # kernel_ is the kernel after applying gp.fit(points, data)
    # The gp stores the hyperparameters as theta = ln(hyper_pramas)
    hp = np.exp(gp.kernel_.theta)
    # The scale and noise terms in gp are sigma_f^2 and sigma_n^2.
    # You want sigma_f and sigma_n.
    hp[0] = hp[0]**0.5
    hp[-1] = hp[-1]**0.5
    return hp


def reasonable_hyperparameters_range(data, limits, sigma_f_factor=[0.1, 4.0], sigma_n_factor=[1.0e-5, 0.1],
                                     length_scale_factor=[0.1, 4.0]):
    """Guesses for the range of hyperparameter values.

    sigma_f [at least, most] sigma_f_factor times the largest deviation of data from zero.
    sigma_n [at least, most] sigma_n_factor times the largest deviation of data from zero.
    length scales [at least, most] length_scale_factor times the range of values in each dimension.
    """
    data_max = np.abs(data).max()

    sigma_f_range = [[sigma_f_factor[0]*data_max, sigma_f_factor[1]*data_max]]
    sigma_n_range = [[sigma_n_factor[0]*data_max, sigma_n_factor[1]*data_max]]

    length_scale_ranges = []
    for i in range(len(limits)):
        ximin, ximax = limits[i, 0], limits[i, 1]
        interval = ximax - ximin
        li_range = [length_scale_factor[0]*interval, length_scale_factor[1]*interval]
        length_scale_ranges.append(li_range)

    hyper_limits = np.array(sigma_f_range + length_scale_ranges + sigma_n_range)
    # Arithmetic average:
    #hp0 = np.array([(hyper_limits[i, 0]+hyper_limits[i, 1])/2.0 for i in range(len(hyper_limits))])
    # Geometric average:
    hp0 = np.array([(hyper_limits[i, 0]*hyper_limits[i, 1])**(1.0/2.0) for i in range(len(hyper_limits))])
    return hp0, hyper_limits


def save_gaussian_process_regression_list(filename, gp_list, kernel_type):
    """
    Save the information needed to reconstruct a list of
    GaussianProcessRegressor objects as an hdf5 file.

    Parameters
    ----------
    filename : string
        hdf5 filename
    gp_list : List of GaussianProcessRegressor objects
    kernel_type : 'squaredexponential', 'matern32', 'matern52'
    """
    f = h5py.File(filename)

    ngp = len(gp_list)
    for i in range(ngp):
        # Create group for each GaussianProcessRegressor object
        groupname = 'gp_'+str(i)
        group = f.create_group(groupname)
        # write necessary data to reconstruct gp
        gp = gp_list[i]
        #group['kernel_type'] = [kernel_type]
        # a single string can't be stored as a data set
        # (although you could make it a single element list).
        # Store the string as an attribute instead.
        group.attrs['kernel_type'] = kernel_type
        group['hyperparameters'] = get_hyperparameters(gp)
        group['points'] = gp.X_train_
        group['data'] = gp.y_train_

    f.close()


def load_gaussian_process_regression_list(filename):
    """
    Load a list of GaussianProcessRegressor objects from an hdf5 file.

    Parameters
    ----------
    filename : string
        hdf5 filename

    Returns
    -------
    gp_list : List of GaussianProcessRegressor objects
    """
    f = h5py.File(filename)
    groups = f.keys()
    ngp = len(groups)

    gp_list = []
    for i in range(ngp):
        groupname = 'gp_'+str(i)
        #kernel_type = f[groupname]['kernel_type'][:]
        kernel_type = f[groupname].attrs['kernel_type']
        hp0 = f[groupname]['hyperparameters']
        points = f[groupname]['points']
        data = f[groupname]['data']
        gp = generate_gp(points, data, hp0, fixed=True, kernel_type=kernel_type)
        gp_list.append(gp)

    f.close()

    return gp_list


#####################     Selecting new points with GPR    #####################


def sample_new_point_with_fixed_hyperparameters(points, limits, hp0, kernel_type, nsamples=100000):
    """Get new point using uncertainty sampling.
    The points are needed but the data is not, because the GPR uncertainty
    does not depend on the data.

    Parameters
    ----------
    points : 2d array (npoints, ndim).
        Coordinates of sampled data.
    limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
        Boundaries of the hypercube.
    hp0 : 1d array (ndim+2,)
        Initial hyperparameter guess for optimizer.
        Order is (sigma_f, ls_0, ls_1, ..., sigma_n).
    kernel_type : 'squaredexponential', 'matern32', 'matern52'
    nsamples : int
        number of random points to search for largest GPR uncertainty.

    Returns
    -------
    point_new : The new point
    gp : GaussianProcessRegressor
        The gpr used for generating this point.
    """
    # Make fake data
    data = np.ones(len(points))

    # Approximate the response surface with GPR
    gp = generate_gp(points, data, hp0, fixed=True, kernel_type=kernel_type)

    # ******* Perform uncertainty sampling *******
    # --Find the point with the maximum uncertainty given the current Gaussian process.
    # --For now use Monte Carlo sampling.
    # --Later, do a multistart optimization.
    test_points = doe.uniform_random_samples(nsamples, limits)
    # Do you want to use absolute error or fractional error?
    # This is where you code up custom objective function to optimize.
    test_errs = gp.predict(test_points, return_std=True)[1]
    i_max = np.argmax(test_errs)
    point_new = test_points[i_max]

    #print np.max(test_errs)

    return point_new, gp


def sample_n_new_points_with_fixed_hyperparameters(n_new, points, limits, hp0, kernel_type, nsamples=100000):
    """Get n_new new points.
    """
    points_updated = copy.copy(points)

    for i in range(n_new):
        print i,
        # Find the best location for the new point
        point_new, gp = sample_new_point_with_fixed_hyperparameters(points_updated, limits, hp0, kernel_type,
                                                                    nsamples=nsamples)

        # Add the new point to the list of updated points
        points_updated = np.concatenate((points_updated, np.atleast_2d(point_new)))

    # Return just the new points that you added
    return points_updated[-n_new:]


def sample_new_point_with_new_data(points, data, limits, hp0, kernel_type, hyper_limits,
                                   n_restarts_optimizer=9, nsamples=100000):
    """Get new point.
    """
    # Approximate response surface with Gaussian process
    gp = generate_gp(points, data, hp0, kernel_type=kernel_type, hyper_limits=hyper_limits,
                     n_restarts_optimizer=n_restarts_optimizer)

    # ******* Perform uncertainty sampling *******
    # --Find the point with the maximum uncertainty given the current Gaussian process.
    # --For now use Monte Carlo sampling.
    # --Later, do a multistart optimization.
    test_points = doe.uniform_random_samples(nsamples, limits)
    # Do you want to use absolute error or fractional error?
    # This is where you code up custom objective function to optimize.
    test_errs = gp.predict(test_points, return_std=True)[1]
    i_max = np.argmax(test_errs)
    point_new = test_points[i_max]

    #print np.max(test_errs)

    return point_new, gp


# #################################### Grids #####################################
#
# def uniform_random_samples(ndata, limits):
#     """Select parameters in a hypercube sampled uniformly in the range limits.
#
#     Parameters
#     ----------
#     Ndata : int
#         Number of points to select.
#     limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
#         Boundaries of the hypercube.
#     """
#     ndim = len(limits)
#
#     param_columns = []
#     for i in range(ndim):
#         xi_min, xi_max = limits[i]
#         xis = np.random.uniform(low=xi_min, high=xi_max, size=ndata)
#         param_columns.append(xis)
#
#     return np.array(param_columns).T
#
#
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
#
#
# ########################## Wrappers for scikit-learn ###########################
#
#
# def generate_square_exponential_gp(params, data, hp0, limits, n_restarts_optimizer=9):
#     """Gaussian Process for n dimensional set of data.
#
#     Parameters
#     ----------
#     params : array of shape (nparams, ndim).
#         Parameters of sampled data.
#     data : array of shape (nparams,).
#         Data at each of the sampled parameters.
#     hp0 : array of shape (ndim+2,)
#         Initial hyperparameter guess for optimizer.
#         Order is (sigma_f, ls_0, ls_1, ..., sigma_n).
#     limits : array of shape (ndim, 2)
#         Lower and upper bounds on the value of each hyperparameter.
#     n_restarts_optimizer : int
#         Number of random points in the hyperparameter space to restart optimization
#         routine for searching for the maximum log-likelihood.
#     """
#     # The constant term is sigma_f**2, but the parameter that you want to give it is sigma_f
#     const = ConstantKernel(hp0[0]**2, limits[0]**2)
#     # Length scales l_i (not l_i**2)
#     sqexp = RBF(hp0[1:-1], limits[1:-1])
#     # Noise term sigma_n (not sigma_n**2)
#     noise = WhiteKernel(hp0[-1], limits[-1])
#     kernel = const*sqexp + noise
#
#     # Initialize the Gaussian Process
#     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
#
#     # Optimize the hyperparameters by maximizing the log likelihood
#     gp.fit(params, data)
#
#     return gp




################################################################################
################################################################################
#                                                                              #
# Ben's custom code for learning how GPR works.                                #
# Don't use this for real work.                                                #
#                                                                              #
################################################################################
################################################################################

class GaussianProcess(object):
    """
    """
    def __init__(self, params, y, cov, hyperparams):
        """Gaussian Process for n dimensional set of data.

        Parameters
        ----------
        params : array of shape (nparams, ndim).
            Parameters of sampled data.
        y : array of shape nparams.
            Data at each of the sampled parameters.
        cov : func(p1, p2, hyperparams)
            Covariance function.
        hyperparams : args
            Hyperparameters for the covariance function.
        """
        self.params = params
        self.y = y
        self.cov = cov
        self.hyperparams = hyperparams

        self.nparams, self.ndim = params.shape

        # Evaluate K
        self.K = np.array([[self.cov(p1, p2, self.hyperparams) for p2 in self.params] for p1 in self.params])

        # Evaluate K^{-1}
        self.Kinv = np.linalg.inv(self.K)

        # Evaluate (K^{-1})_{ij} y_j (array of length nparams).
        self.Kinv_dot_y = np.dot(self.Kinv, self.y)

    def __call__(self, pst):
        """Interpolate the data at the point pst.

        Parameters
        ----------
        pst : array of shape ndim.
            Point p_* where you want to evaluate the function.

        Returns
        -------
        yst : float
            Interpolated value at the point pst.
        """
        # Evaluate vector K_*
        Kst = np.array([self.cov(pst, p1, self.hyperparams) for p1 in self.params])

        # Evaluate y_*
        return np.dot(Kst, self.Kinv_dot_y)

    def error(self, pst):
        """Estimate the interpolation error at the point pst.

        Parameters
        ----------
        pst : array of shape ndim.
            Point p_* where you want to evaluate the function.

        Returns
        -------
        sqrt(yst_var) : float
            Estimate of the interpolation error at point pst.
        """
        # Evaluate vector K_* and point K_**
        Kst = np.array([self.cov(pst, p1, self.hyperparams) for p1 in self.params])
        Kstst = self.cov(pst, pst, self.hyperparams)

        # Evaluate variance at y_*
        yst_var = Kstst - np.dot(Kst, np.dot(self.Kinv, Kst.T) )
        return np.sqrt(yst_var)

    def ln_like(self):
        """ln(likelihood) of the Gaussian Process given the hyperparameters covargs.
        Expression is the sum of 3 terms: data, complexity penalty, normalization
        """
        Kdet = np.linalg.det(self.K)
        return -0.5*np.dot(self.y.T, self.Kinv_dot_y) - 0.5*np.log(Kdet) - 0.5*self.nparams*np.log(2.0*np.pi)


############ GPR Set class that allows you to vary hyperparameters #############

# Class for set of GaussianProcess with common covariance function and data
# Should this be a class that inherits the methods of GaussianProcess? Is that how things work?
class GaussianProcessSet(object):
    def __init__(self, params, y, cov):
        """Gaussian Process for n dimensional set of data.

        Parameters
        ----------
        params : array of shape (nparams, ndim).
            Parameters of sampled data.
        y : array of shape nparams.
            Data at each of the sampled parameters.
        cov : func(p1, p2, hyperparams)
            Covariance function.
        """
        self.cov = cov
        self.params = params
        self.y = y

    def ln_like_of_log_hyperparams(self, log10_hyperparams):
        """ln(likelihood) as a function of log10(hyperparameters).
        """
        gp = GaussianProcess(self.params, self.y, self.cov, 10.0**log10_hyperparams)
        return gp.ln_like()

    def neg_ln_like(self, log10_hyperparams):
        """-ln(likelihood) as a function of log10(hyperparameters).
        Used in optimization routines which always do minimization instead of maximization.
        """
        return -self.ln_like_of_log_hyperparams(log10_hyperparams)

    def optimize(self, log10_hp0):
        """
        Parameters
        ----------
        log10_hp0 : array shape nhyperparams
            Initial guess for optimal log10(hyperparams).
        """
        res = optimize.minimize(self.neg_ln_like, log10_hp0)
        return 10**res.x


########################### Covariance functions ###############################

def squared_exponential_covariance(p1, p2, hyperparams):
    """Squared exponential covariance function for n-dimensional data.

    Parameters
    ----------
    p1 : array with shape ndim
    p2 : array with shape ndim
    hyperparams : array with shape ndim+2 [sigma_f, sigma_n, ls0, ls1, ...]
        sigma_f : Approximately the range (ymax-ymin) of values that the data takes.
            sigma_f^2 called the signal variance.
        sigma_n : Noise term. The uncertainty in the y values of the data.
        ls0 : Length scales for the variation in dimension 0.

    Returns
    -------
    covariance : float
    """
    sigma_f = hyperparams[0]
    sigma_n = hyperparams[1]
    ls = hyperparams[2:]
    ndim = len(ls)
    # !!!!!!!! assert that len(p1) == len(p2) == len(hyperparams)-2 !!!!!!!! #

    # Noise nugget for diagonal elements
    if np.array_equal(p1, p2):
        nugget = sigma_n**2
    else:
        nugget = 0.0

    # Calculate the covariance
    squares = np.array([-(p1[i]-p2[i])**2 / (2*ls[i]**2) for i in range(ndim)])
    return sigma_f**2 * np.exp(np.sum(squares)) + nugget
