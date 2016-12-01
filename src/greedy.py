import numpy as np


################################################################################
#                   Generate reduced basis with greedy method                  #
################################################################################

def projection(h_new, orthonorm_list, add, scalar_multiply, inner_product):
    """Calculate Proj(h_new) = sum c_i * e_i,
    where e_i = orthonorm_list[i],
    and c_i = < hnew | e_i >.
    This is the projection of a waveform hnew onto a vector space
    formed by the *orthonormal* basis orthonorm_list.
    
    Parameters
    ----------
    orthonorm_list : List of basis vector waveforms.
        Set of *orthogonal* and *normalized* waveforms.
    h_new :
        A New waveform.
    add : function(h1, h2)
        Function that adds two waveforms.
    scalar_multiply : function(alpha, h)
        Function that multiplies the waveform h by the scalar alpha.
    inner_product : function(h1, h2)
        Function that evaluates the inner product (scalar) of two waveforms.
        
    Returns
    -------
    proj_onto_span :
        Proj(h_new).
    """
    # Dimension of subspace spanned by the vectors orthonorm_list
    Nbases = len(orthonorm_list)
    
    # Coefficients for inner product between hnew and the other waveforms
    # c_i = < hnew | hnorm_i >
    coeff_vec = np.array([inner_product(orthonorm_list[i], h_new) for i in range(Nbases)])
    
    # Calculate projection of hnew onto span(orthonorm_list)
    # Allocate memory for the new waveform
    # !!!!!!!! This could be replaced by a function called matrix_vector_multiply !!!!!!!! if this is too slow.
    proj_onto_span = scalar_multiply(coeff_vec[0], orthonorm_list[0])
    for i in range(1, Nbases):
        # Accumulate: proj = proj + c_i*e_i
        proj_onto_span = add(proj_onto_span, scalar_multiply(coeff_vec[i], orthonorm_list[i]))
    
    return proj_onto_span


def subtract_projection(h_new, orthonorm_list, add, subtract, scalar_multiply, inner_product):
    """Calculate h_new - Proj(h_new):
    The component of hnew orthogonal to the
    projection of hnew onto the vector space
    formed by the *orthonormal* basis orthonorm_list.
        
    Parameters
    ----------
    orthonorm_list : List of basis vector waveforms.
        Set of *orthogonal* and *normalized* waveforms.
    h_new :
        A New waveform.
    add : function(h1, h2)
        Function that adds two waveforms.
    subtract : function(h1, h2)
        Function that subtracts two waveforms (h1-h2).
    scalar_multiply : function(alpha, h)
        Function that multiplies the waveform h by the scalar alpha.
    inner_product : function(h1, h2)
        Function that evaluates the inner product (scalar) of two waveforms.
    
    Returns
    -------
    diff :
        h_new - Proj(h_new)
    """
    proj_onto_span = projection(h_new, orthonorm_list, add, scalar_multiply, inner_product)
    return subtract(h_new, proj_onto_span)

def projection_from_precalculated_inner_products(h_new, orthonorm_list, add, scalar_multiply, inner_product_list):
    """Calculate Proj(h_new) = sum c_i * e_i,
    where e_i = orthonorm_list[i],
    and c_i = < hnew | e_i >.
    This is the projection of a waveform hnew onto a vector space
    formed by the *orthonormal* basis orthonorm_list.
    
    Parameters
    ----------
    orthonorm_list : List of basis vector waveforms.
        Set of *orthogonal* and *normalized* waveforms.
    h_new :
        A New waveform.
    add : function(h1, h2)
        Function that adds two waveforms.
    scalar_multiply : function(alpha, h)
        Function that multiplies the waveform h by the scalar alpha.
    inner_product_list : array
        List of inner products between h_new and orthonorm_list.
        
    Returns
    -------
    proj_onto_span :
        Proj(h_new).
    """
    # Dimension of subspace spanned by the vectors orthonorm_list
    Nbases = len(orthonorm_list)
    
    # Coefficients for inner product between hnew and the other waveforms
    # c_i = < hnew | hnorm_i >
    coeff_vec = inner_product_list
    
    # Calculate projection of hnew onto span(orthonorm_list)
    # Allocate memory for the new waveform
    # !!!!!!!! This could be replaced by a function called matrix_vector_multiply !!!!!!!! if this is too slow.
    proj_onto_span = scalar_multiply(coeff_vec[0], orthonorm_list[0])
    for i in range(1, Nbases):
        # Accumulate: proj = proj + c_i*e_i
        proj_onto_span = add(proj_onto_span, scalar_multiply(coeff_vec[i], orthonorm_list[i]))
    
    return proj_onto_span


def subtract_projection_from_precalculated_inner_products(h_new, orthonorm_list, add, subtract, scalar_multiply, inner_product_list):
    """Calculate h_new - Proj(h_new):
    The component of hnew orthogonal to the
    projection of hnew onto the vector space
    formed by the *orthonormal* basis orthonorm_list.
        
    Parameters
    ----------
    orthonorm_list : List of basis vector waveforms.
        Set of *orthogonal* and *normalized* waveforms.
    h_new :
        A New waveform.
    add : function(h1, h2)
        Function that adds two waveforms.
    subtract : function(h1, h2)
        Function that subtracts two waveforms (h1-h2).
    scalar_multiply : function(alpha, h)
        Function that multiplies the waveform h by the scalar alpha.
    inner_product_list : array
        List of inner products between h_new and orthonorm_list.
    
    Returns
    -------
    diff :
        h_new - Proj(h_new)
    """
    proj_onto_span = projection_from_precalculated_inner_products(h_new, orthonorm_list, add, scalar_multiply, inner_product_list)
    return subtract(h_new, proj_onto_span)


def add_basis_with_iterated_modified_gram_schmidt(h, basis, add, subtract, scalar_multiply, inner_product, a=0.5, max_iter=3):
    """Given a function h, find the corresponding basis function orthonormal to all previous ones.
    
    Parameters
    ----------
    h : waveform
        basis : List of *orthonormal* waveforms
    a : int
        Parameter to determine if the Gram-Schmidt procedure was made the new basis sufficiently orthogonal
    max_iter : int
        Number of times to iterate the Gram-Schmidt procedure
    """
    # the norm of h and the normalized version e_new
    # The np.abs() just gets rid of the not-quite-numerically-zero imaginary part)
    norm = np.sqrt(np.abs(inner_product(h, h)))
    e_new = scalar_multiply(1.0/norm, h)
    
    # flag = 1 when Gram-Schmidt has been done enough times
    flag = 0
    # ctr is the counter for the number of iterations
    ctr = 1
    while flag == 0:
        ##### Start Gram-Schmidt orthogonalization #####
        # Subtract Proj_{basis}(e_new) (the projection of enew onto the basis)
        e_new = subtract_projection(e_new, basis, add, subtract, scalar_multiply, inner_product)
        
        # Get the norm of the remainder, but don't renormalize enew yet
        new_norm = np.sqrt(np.abs(inner_product(e_new, e_new)))
        ##### End Gram-Schmidt #####
        
        # Determine if you need to do another iteration of Gram-Schmidt.
        # if h was almost parallel to span(basis), then the Gram-Schmidt process involved
        # subtracting two nearly equal vectors. This leads to a catastrophic loss in precission.
        # You can determine if enew and Proj_{basis}(enew) were nearly equal by comparing norm and new_norm.
        # If new_norm is << norm, then the orthogonal component is small, and they were nearly equal.
        if new_norm/norm <= a:
            # Repeat the while loop and
            # continue subtracting Proj_{basis}(enew) until enew is truly orthogonal to span(basis)
            norm = new_norm
            ctr += 1
            if ctr > max_iter:
                print "Max number of iterations ("+str(max_iter)+") reached in iterated Gram-Schmidt. Basis may not be orthonormal."
        else:
            flag = 1
    
    # Return the new ortho*normal* basis function
    return scalar_multiply(1.0/new_norm, e_new)


class ReducedBasis(object):
    """Functions for generating a set of reduced basis waveforms with the greedy algorithm.
    
    Attributes
    ----------
    add_func
    subtract_func
    scalar_multiply_func
    inner_product_func
    get_waveform
    ts_params
    norms
    inner_products
    sigma_list
    rb
    rb_indices
    rb_params
    """
    
    #def __init__(self, add_func, subtract_func, scalar_multiply_func, inner_product_func, abs_max_func, get_waveform, ts_params):
    def __init__(self, add_func, subtract_func, scalar_multiply_func, inner_product_func, get_waveform, ts_params):
        """Create an empty ReducedBasis object.
        Define functions for addition, subtraction, scalar multiplication, inner product of waveforms.
        
        Parameters
        ----------
        add : function(h1, h2)
            Function that adds two waveforms.
        subtract : function(h1, h2)
            Function that subtracts two waveforms (h1-h2).
        scalar_multiply : function(alpha, h)
            Function that multiplies the waveform h by the scalar alpha.
        inner_product : function(h1, h2)
            Function that evaluates the inner product (scalar) of two waveforms.
        get_waveform : function(i)
            Function that fetches the waveform in the TrainingSet with index i
        ts_params : 2d list
            List of the waveform parameters
        """
        # The waveform functions
        self.add_func = add_func
        self.subtract_func = subtract_func
        self.scalar_multiply_func = scalar_multiply_func
        self.inner_product_func = inner_product_func
        #self.abs_max_func = abs_max_func
        self.get_waveform = get_waveform
        self.ts_params=ts_params
        
        # The norm sqrt(<hi|hi>) for the training set
        self.norms = None
        
        # List of inner products between the current reduced basis and the training set
        # Has shape N_RB X N_TS
        self.inner_products = None
        
        self.sigma_list = None # Greedy error
        self.rb = None # List of reduced basis waveforms
        self.rb_indices = None # Indices corresponding to the training set waveforms used to construct the RB
        self.rb_params = None # Waveform parameters corresponding to the training set waveforms rb_indices
    
    def normalize_waveform(self, h):
        """Normalize the waveform h.
        
        Returns
        -------
        norm : float
        h_norm : The normalized waveform.
        """
        # The np.abs() just gets rid of the not-quite-numerically-zero imaginary part)
        norm = np.sqrt(np.abs(self.inner_product_func(h, h)))
        return norm, self.scalar_multiply_func(1.0/norm, h)
    
    def calculate_norms(self):
        """Calculate the norms for each waveform in the training set
        """
        
        # The np.abs() just gets rid of the not-quite-numerically-zero imaginary part)
        Nts = len(self.ts_params)
        self.norms = [np.sqrt(np.abs(self.inner_product_func(self.get_waveform(i), self.get_waveform(i)))) for i in range(Nts)]
    
    def set_initial_rb(self, ts_index):
        """Set the initial RB waveform to be the one from the training set
        with index ts_index.
        
        Parameters
        ----------
        ts_istart : int
            Index for the first RB waveform. 0th waveform in training set by default.
        """
        self.inner_products = []
        self.sigma_list = [1.0]
        self.rb_indices = [ts_index]
        self.rb_params = [self.ts_params[ts_index]]
        self.rb = [self.normalize_waveform(self.get_waveform(ts_index))[1]]
    
    def _append_inner_products(self):
        """Calculate the inner products between the training set waveforms h_i
        and the newest RB waveform e_j. Then, append them to the inner_products list.
        """
        Nts = len(self.ts_params)
        Nrb = len(self.rb)
        
        h_i_dot_e_j = [self.inner_product_func(self.get_waveform(i), self.rb[-1]) for i in range(Nts)]
        self.inner_products.append(h_i_dot_e_j)
    
    def _get_distances_from_current_rb(self):
        """Calculate the orthogonal distance between each member of the training set and the current RB.
        Do this efficiently by using the already stored inner_products matrix,
        so that you only have to calculate the inner products with the newest RB waveform e_j.
        
        Returns
        -------
        distances : list of length Nts
        """
        
        Nts = len(self.ts_params)
        Nrb = len(self.rb)
        
        # Calculate the inner products between the training set and the last RB waveform e_j.
        self._append_inner_products()
        
        # Calculate the distance between each training set
        distances = [1.0 - np.sum([np.abs(self.inner_products[j][i])**2 for j in range(Nrb)])/self.norms[i]**2
                     for i in range(Nts)]
        return distances
    
#    def _get_distances_from_current_rb(self):
#        """Calculate the orthogonal distance between each member of the training set and the current RB.
#        Do this efficiently by using the already stored inner_products matrix,
#        so that you only have to calculate the inner products with the newest RB waveform e_j.
#        
#        Returns
#        -------
#        distances : list of length Nts
#        """
#        
#        Nts = len(self.ts_params)
#        Nrb = len(self.rb)
#        
#        # Calculate the inner products between the training set and the last RB waveform e_j.
#        self._append_inner_products()
#
#        # Calculate the distance between each training set
#        distances = []
#        for i in range(Nts):
#            h_new = self.get_waveform(i)
#            inner_product_list = np.array(self.inner_products)[:, i]
#            difference = subtract_projection_from_precalculated_inner_products(h_new, 
#                self.rb, self.add_func, self.subtract_func, self.scalar_multiply_func, inner_product_list)
#            # Calculate the L_infty norm             
#            abs_max_distance = self.abs_max_func(difference)/self.abs_max_func(h_new)
#            distances.append(abs_max_distance)
#
#        return distances

    def generate_new_basis_from_training_set(self):
        """Generate the next RB waveform from the most orthogonal member of the training set.
        """
        # Find the most orthogonal waveform in the training_set h_new
        distances = self._get_distances_from_current_rb()
        i_new = np.argmax(distances)
        self.sigma_list.append(distances[i_new])
        self.rb_indices.append(i_new)
        self.rb_params.append(self.ts_params[i_new])
        
        # Generate the new RB element with the Gram-Schmidt orthonormalization process:
        # e_(j+1) = (h_new - Proj_{e_1...e_j}(h_new))/norm(h_new)
        h_new = self.get_waveform(i_new)
        e_new = add_basis_with_iterated_modified_gram_schmidt(h_new, self.rb,
                                                              self.add_func, self.subtract_func,
                                                              self.scalar_multiply_func, self.inner_product_func,
                                                              a=0.5, max_iter=3)
        self.rb.append(e_new)
    
    def generate_new_basis_from_specific_waveform(self, ts_index):
        """Force the next RB waveform to be constructed from the waveform
        in the training set with the index ts_index.
        
        Also, evaluates the inner product for the remaining training set waveforms,
        since you will need them for later RB waveforms.
        
        Parameters
        ----------
        ts_index : int
            Index of the training set waveform you want to use.
        """
        # Calculate the inner products between the training set and the last RB waveform e_j.
        #self._append_inner_products()
        
        # The greedy error is still the greatest distance over the entire training set
        # indexed by i_new, even though you choose the specific waveform ts[ts_index].
        distances = self._get_distances_from_current_rb()
        i_new = np.argmax(distances)
        self.sigma_list.append(distances[i_new])
        
        # Set the index and parameters for the new RB waveform
        self.rb_indices.append(ts_index)
        self.rb_params.append(self.ts_params[ts_index])
        
        # Generate the new RB element with the Gram-Schmidt orthonormalization process:
        # e_(j+1) = (h_new - Proj_{e_1...e_j}(h_new))/norm(h_new)
        h_new = self.get_waveform(ts_index)
        e_new = add_basis_with_iterated_modified_gram_schmidt(h_new, self.rb,
                                                              self.add_func, self.subtract_func,
                                                              self.scalar_multiply_func, self.inner_product_func,
                                                              a=0.5, max_iter=3)
        self.rb.append(e_new)
    
    def generate_reduced_basis(self, epsilon, ts_istart=0, Nbases=None):
        """Generate an orthonormal set of waveforms with the greedy algorithm.
        
        Parameters
        ----------
        epsilon : float
            The tolerance. Should be greater than 1.0e-14.
        ts_istart : int
            Index for the first RB waveform. 0th waveform in training set by default.
        Nbases : int
            Maximum number of bases to generate.
        """
        # Maximum number of bases
        if Nbases == None:
            Nbases = len(self.ts_params)
        
        # Calculate the norms of all the training set waveforms
        self.calculate_norms()
        
        # Set the first RB waveform to be the (normalized) 1st element in the training set
        self.set_initial_rb(ts_istart)
        print self.sigma_list[-1]
        
        # Add new RB waveforms until the greedy error is < epsilon
        while self.sigma_list[-1]>epsilon and len(self.rb)<Nbases:
            self.generate_new_basis_from_training_set()
            print self.sigma_list[-1]
    
    def generate_reduced_basis_start_with_set(self, epsilon, ts_initial_list, Nbases=None):
        """Generate an orthonormal set of waveforms with the greedy algorithm.
        the first waveforms in the RB are constructed from ts_initial_list
        
        Parameters
        ----------
        epsilon : float
            The tolerance. Should be greater than 1.0e-14.
        ts_initial_list : int
            Indices of the waveforms required to be in the RB.
        Nbases : int
            Maximum number of bases to generate.
        """
        # Maximum number of bases
        if Nbases == None:
            Nbases = len(self.ts_params)
        
        # Calculate the norms of all the training set waveforms
        self.calculate_norms()
        
        # Set the first RB waveform
        print 'Using the requested waveforms.'
        self.set_initial_rb(ts_initial_list[0])
        print self.sigma_list[-1]
        
        # Add new RB waveforms from the list ts_initial_list
        for i in range(1, len(ts_initial_list)):
            self.generate_new_basis_from_specific_waveform(ts_initial_list[i])
            print self.sigma_list[-1]
        
        # Add new RB waveforms until the greedy error is < epsilon
        print 'Now choosing waveforms with the greedy method.'
        while self.sigma_list[-1]>epsilon and len(self.rb)<Nbases:
            self.generate_new_basis_from_training_set()
            print self.sigma_list[-1]

