import numpy as np
import scipy.optimize as optimize

#################################### Grids #####################################

def uniform_random_samples(ndata, limits):
    """Select parameters in a hypercube sampled uniformly in the range limits.
    
    Parameters
    ----------
    Ndata : int
        Number of points to select.
    limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
        Boundaries of the hypercube.
    """
    ndim = len(limits)
    
    param_columns = []
    for i in range(ndim):
        xi_min, xi_max = limits[i]
        xis = np.random.uniform(low=xi_min, high=xi_max, size=ndata)
        param_columns.append(xis)
    
    return np.array(param_columns).T


def latin_hypercube(Ndata, limits):
    """Select parameters using the Latin Hypercube method.
    
    Parameters
    ----------
    Ndata : int
        Number of points to select.
    limits : np.array([[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max], ...])
        Boundaries of the hypercube.
    """
    Nparams = len(limits)
    
    # Find indices of grid points
    point_ind_list = []
    allowed_indices = np.array([[n for n in range(Ndata)] for i in range(Nparams)])
    for n in range(Ndata):
        js = np.random.randint(Ndata-n, size=Nparams)
        point_ind = [allowed_indices[i, js[i]] for i in range(Nparams)]
        point_ind_list.append(point_ind)
        allowed_indices = np.array([np.delete(allowed_indices[i], js[i]) for i in range(Nparams)])
    
    point_ind_array = np.array(point_ind_list)
    
    # Get grid points
    grids = []
    for i in range(Nparams):
        grids.append(np.linspace(limits[i, 0], limits[i, 1], Ndata))
    
    grids_array = np.array(grids)
    
    # Get chosen points on the grid
    points = np.array([[grids_array[i, point_ind_array[n, i]] for i in range(Nparams)] for n in range(Ndata)])
    
    return points


##################### GPR class ########################

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


