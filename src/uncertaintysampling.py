from time import time
import numpy as np
import scipy.optimize as optimize
import gaussianprocessregression as gpr


def find_minimum(func, limits, nbasinjumps=20, nfun_eval_per_basin=15):
    """Find the minimum of func within the range given by limits.
    Uses basinhopping to jump between local minima, and 'L-BFGS-B' to find local minimum.
    The number of function evaluations will be approximately nbasinjumps*nfun_eval_per_basin.
    Features are rescaled from x\in[a, b] to xi\in[0, 1] before minimization.

    Parameters
    ----------
    func(params) : Objective function that takes a 1d-array of parameters.
    limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
    nbasinjumps : int
        Number of times to search for a new local minimum.
    nfun_eval_per_basin : int
        Approximate max number of evaluations of func per local minimum.

    Returns
    -------
    x_min : 1d-array
        Minimum (hopefully global minimum).
    f_min : float
        Function value at minimum
    points : 2d-array of shape (n_eval, nparams)
        The points the function was evaluated at.
    i : int
        Number of function evaluations.
    """
    ######### Rescaling #########

    nparams = len(limits)
    xi_limits = np.array([[0., 1.]]*nparams)

    # Initial guess uniformly distributed in parameter space
    xi0 = np.random.uniform(low=0.0, high=1.0, size=nparams)

    points = []
    def func_scaled(xi):
        """Takes parameters from [0, 1] in each coordinate.
        """
        # Counter for function evaluations
        func_scaled.i += 1
        # Rescale each parameter to [a, b]
        x = np.array([(limits[i, 1]-limits[i, 0])*xi[i] + limits[i, 0] for i in range(nparams)])
        points.append(x)
        return func(x)

    def xi_boundary(**kwargs):
        """Return true or false.
        """
        xi = kwargs['x_new']
        # Test if xi is above 0
        tmin = bool(np.all(xi >= 0.0))
        # Test if xi is below 1
        tmax = bool(np.all(xi <= 1.0))
        # Test if xi is inside [0, 1]
        inside = tmin and tmax
        return inside

    ######### Minimize func_scaled(xi) #########

    options={'maxfun':nfun_eval_per_basin}
    minimizer_kwargs = {'method':'L-BFGS-B', 'bounds':xi_limits, 'options':options}
    func_scaled.i = 0
    ret = optimize.basinhopping(
        func_scaled, xi0, niter=nbasinjumps, accept_test=xi_boundary,
        minimizer_kwargs=minimizer_kwargs)

    ######### Convert minimum xi_min back to unscaled units x_min ########
    f_min = ret.fun
    xi_min = ret.x
    x_min = np.array([(limits[i, 1]-limits[i, 0])*xi_min[i] + limits[i, 0] for i in range(nparams)])


    return x_min, f_min, np.array(points), func_scaled.i


def rms_phase_error(point, sigma_dphase_gp_list):
    """Objective function to *MAXIMIZE* when searching for new training set points.
    This is the root-mean-squared error estimate of the phase at the empirical nodes F^phi_j.
    """
    nphase = len(sigma_dphase_gp_list)

    # Calculate waveform at nodes
    sigma_dphase = np.array([sigma_dphase_gp_list[j].predict(np.atleast_2d(point), return_std=True)[1][0]
                               for j in range(nphase)])

    return np.sqrt(np.sum(sigma_dphase**2)/nphase)


##################### Class for performing uncertainty sampling ################

class UncertaintySampling(object):

    def __init__(self, original_points, limits, kernel_type, dphase_gp_list):
        # Quantities that don't change
        self.original_points = original_points
        self.limits = limits
        self.kernel_type = kernel_type
        self.original_dphase_gp_list = dphase_gp_list
        self.nphase = len(self.original_dphase_gp_list)

        # Quantities that change
        self.sigma_dphase_list = None
        self.new_points = None
        self.new_errors = None

    def update_uncertainties(self):
        """Update the list of GPR functions for evaluating
        DeltaPhi(f; x) at the empirical nodes F^phi_j.
        This uses all available points (original_points+new_points).
        """
        if self.new_points is None:
            points = self.original_points
        else:
            points = np.concatenate((self.original_points, self.new_points))

        # Make fake data
        # (With fixed hyperparameters GPR error estimate doesn't depend on data.)
        data = np.ones(len(points))

        # Generate GPR at each phase node
        self.sigma_dphase_list = []
        for j in range(self.nphase):
            hyper = gpr.get_hyperparameters(self.original_dphase_gp_list[j])
            gp = gpr.generate_gp(points, data, hyper, fixed=True, kernel_type=self.kernel_type)
            self.sigma_dphase_list.append(gp)

    def negative_error(self, params):
        """The function you want to minimize.
        """
        neg_err = -rms_phase_error(params, self.sigma_dphase_list)
        return neg_err

    def maximize_error(self, nbasinjumps=20, nfun_eval_per_basin=15, verbose=True):
        """Maximize the error function by minimizing its negative with
        the basinjumping minimization algorithm.
        """
        t0 = time()

        point_new, fmin, points_eval, neval = find_minimum(
            self.negative_error, self.limits,
            nbasinjumps=nbasinjumps, nfun_eval_per_basin=nfun_eval_per_basin)
        err_new = -fmin

        dt = time() - t0

        if verbose:
            print 'err_new='+str(err_new)+', neval='+str(neval)+', evaluation time='+str(dt)
            print 'point_new='+str(point_new)

        return point_new, err_new

    def add_new_points(self, npoints, **kwargs):
        """Add a new point at the maximum value of the error function.
        """
        for i in range(npoints):
            print i, 
            self.update_uncertainties()
            point_new, err_new = self.maximize_error(**kwargs)

            if self.new_points is None and self.new_errors is None:
                self.new_points = np.array([point_new])
                self.new_errors = np.array(err_new)
            else:
                self.new_points = np.append(self.new_points, [point_new], axis=0)
                self.new_errors = np.append(self.new_errors, err_new)
