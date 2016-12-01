import numpy as np

################################################################################
# Find empirical interpolation nodes, and construct the empirical interpolant.
# These functions are meant to work for any reduced basis that can
# be expressed as a list of array-like objects.
################################################################################

def generate_empirical_nodes(reduced_basis):
    """
    Determine the empirical nodes from the reduced basis.
    
    Parameters
    ----------
    reduced_basis : List of arraylike objects.
        The orthonormalized waveforms making up the reduced basis.
    
    Returns
    -------
    empirical_node_indices : List of integers
        Indices for the waveform at the nodes T_i.
    """
    
    # Get the index of the 0th empirical node T_0
    # from the maximum value of the 0th reduced basis.
    imax = np.argmax(np.abs(reduced_basis[0]))
    empirical_node_indices = [imax]
    
    # Iterate to find the other impirical nodes.
    for j in range(1, len(reduced_basis)):
        # The jth empirical node
        imax = generate_new_empirical_node(reduced_basis, empirical_node_indices)
        empirical_node_indices.append(imax)
    
    return empirical_node_indices


def generate_new_empirical_node(reduced_basis, empirical_node_indices):
    """
    Determine the next empirical node from the current empirical nodes and the reduced basis.
    
    Parameters
    ----------
    reduced_basis : List of arraylike objects.
        The orthonormalized waveforms making up the reduced basis.
    empirical_node_indices : List of integers
        Indices corresponding to the nodes [T_0, ..., T_{j-1}].
    
    Returns
    -------
    imax : int
        Index corresponding to the new node T_j.
        """
    
    # j is current iteration
    j = len(empirical_node_indices)
    
    # Matrix containing A_{ki} = e_i(t_k) =
    # [[e0(T0)    ... e{j-1}(T0)    ]
    #  [          ...               ]
    #  [e0(T{j-1})... e{j-1}(T{j-1})]]
    Aki = np.array([[reduced_basis[i][empirical_node_indices[k]] for i in range(j)] for k in range(j)])
    
    # Vector containing b_k = e_j(T_k) =
    # [ej(T0) ... ej(T{j-1})]
    bk = np.array([reduced_basis[j][empirical_node_indices[k]] for k in range(j)])
    
    # Vector containing C_i =
    # [C0 ... C{j-1}]
    Ci = np.linalg.solve(Aki, bk)
    
    # Evaluate empirical interpolant I_{j-1}[e_j](t) of the basis e_j(t)
    waveformmat = np.array([reduced_basis[i] for i in range(j)])
    interpolant = np.dot(Ci, waveformmat)
    
    # Evaluate residual = I_{j-1}[e_j](t) - e_j(t)
    ej = reduced_basis[j]
    residual = interpolant - ej
    
    # New empirical node is at the maximum of the absolute value of the residual
    imax = np.argmax(np.abs(residual))
    
    # !!!!!! You had previously looked for the argmax of the real part instead of the absolute value !!!!!!!!
    # This is wrong:
    #imax = np.argmax(residual)
    
    return imax


def generate_interpolant_list(reduced_basis, empirical_node_indices):
    """
    Evaluate the TimeSeries B_j(t) that gives the empirical interpolant I_m[h](t) = Sum_{j=1}^m B_j(t) h(T_j)
    when the the quantities h(T_j) are known.
    
    Parameters
    ----------
    reduced_basis : List of arraylike objects.
        The orthonormalized waveforms making up the reduced basis.
    empirical_node_indices : List of integers
        Indices corresponding to the nodes [T_0, ..., T_{j-1}].
    
    Returns
    -------
    B_j: List of 1d arrays
        The interpolating functions
    """
    
    # Dimension of reduced basis
    m = len(reduced_basis)
    
    # Matrix containing V_{ji} = e_i(t_j) =
    # [[e0(T0)    ... e{m-1}(T0)    ]
    #  [          ...               ]
    #  [e0(T{m-1})... e{m-1}(T{m-1})]]
    V_ji = np.array([[reduced_basis[i][empirical_node_indices[j]] for i in range(m)] for j in range(m)])
    
    # Calculate inverse
    Vinverse_ij = np.linalg.inv(V_ji)
    
    # Calculate B_j(t) = Sum_{i=1}^m e_i(t) (Vinv)_ij
    waveformmat_il = np.array([reduced_basis[i] for i in range(m)])
    
    B_j = np.dot(waveformmat_il.T, Vinverse_ij).T
    
    # Convert B_j from 2d array to list of 1d arrays
    return [B_j[j] for j in range(m)]


