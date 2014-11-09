import numpy as np

def proportional_ridge(W, coi, freqs, max_dist=0.05, start_col='max'):
    """proportional_ridge finds a ridge from the cwt that maximizes power

    Parameters
    ----------

    W : ndarray (r x c)
       W is the result of the cwt (can be imaginary)
    max_dist : float
       max_dist is the maximum fraction of the column height that the
       algorithm will search for the next local maxima. 
    coi : ndarray (c,)
       The cone of the influence of the particular wavelet used. Makes
       sure the first peak is within the cone of influence.
    freqs : ndarray (r,)
        Frequencies at which the wavelet transform is computed.
    start_col : "max", "half", or int
        The column with which to start the ridge walking.

    Returns
    -------
    ridge : ndarray(c x 1)
        ridge is the indicies of the ridge found.
    """

    power = np.abs(W) ** 2
    r, c = power.shape

    period = 1./freqs
    coi_inds = np.abs(np.vstack([period]*c).T - coi).argmin(0)

    if start_col == 'max':
        # Mask array where the cwt is influenced by edge effects, get
        # column of the maximum element that is not masked.
        masked_power = np.ma.array(power)
        for i, coi_start in enumerate(coi_inds):
            masked_power[coi_start:, i] = np.ma.masked
        max_index = np.unravel_index(masked_power.argmax(),
                                     masked_power.shape)
        start_col = max_index[1]
            

    if start_col == 'half': start_col = int(c/2)
        

    start_max = power[:coi_inds[start_col],start_col].argmax()
    search_range = int(max_dist*c)  # Number of locations to search

    def find_next_max(curr_max_ind, next_col):
        """ Looks through the next column and returns the index of
        the next local maxima """

        first = curr_max_ind - search_range
        last  = curr_max_ind + search_range
        return next_col[first:last].argmax() + first

    # Pre-allocate ridge array
    ridge = np.zeros(c, dtype=int)
    ridge[start_col] = start_max

    # Second half
    for i in xrange(start_col+1, c):
        ridge[i] = find_next_max(ridge[i - 1], power[:,i])

    # Backwards over first half
    for i in xrange(start_col-1, -1, -1):
        ridge[i] = find_next_max(ridge[i + 1], power[:,i])

    return ridge


# def ridge_extraction(W, T=None, Nparticles=None):
#     """ridge_extraction performs a simulated-annealing ridge location
#     using the algorithm described in Carmona et. al (1999).  Returns a
#     matrix of the size of the field passed to it that contains the mean
#     residence times of all particles at each scale/translation pair.
# 
#     Parameters
#     ----------
# 
#     W : array [r x c]
#        W is the result of the cwt, with r rows and c columns
#     T : array [c x 1]
#        T is the cooling schedule
#     Nparticles : int
#        Nparticles is the number of particles to use
# 
#     Returns
#     -------
#     
#     ridgetable : array [r x c]
#         
# 
#     """
# 
#     W = np.abs(W)
#     r, c = W.shape
# 
#     if T is None:
#         its_per_stage = 4*r
#         T = ([1.000] * its_per_stage + 
#              [0.100] * its_per_stage + 
#              [0.010] * its_per_stage + 
#              [0.001] * its_per_stage)
#         T = np.array(T)
# 
#     MC = np.zeros((r, c))
#     sc = np.floor(np.random.rand(Nparticles) * r) + 1
#     tr = np.floor(np.random.rand(Nparticles) * c) + 1
# 
#     if Nparticles is None:
#         Nparticles = 4*c
# 
#     Niter = len(T)
# 
#     for lc in xrange(Niter):
# 
#         dx = np.sign(np.random.rand(Nparticles) - 0.5)
#         dy = np.sign(np.random.rand(Nparticles) - 0.5)
# 
#         dy[sc == 1] = 1
#         dy[sc == r] = -1
# 
#         proposedA = sc + dy
# 
#         tr = np.mod(tr + dx, c) + 1
# 
#         start_inds = np.array((tr - 1) * r + sc, dtype=int)
#         Wstart = W.flat[start_inds]
#         end_inds = np.array((tr - 1) * r + proposedA, dtype=int)
#         Wend = W.flat[end_inds]
# 
#         DM = Wend - Wstart
# 
#         switch1 = np.random.rand(Nparticles) < np.exp(DM / T[lc])
#         switch2 = DM > 0
#         switchers = np.any(np.array([switch1, switch2]), 0)
#         stayers = ~switchers
# 
#         sc = sum(mcat([switchers, OMPCSEMI, stayers]) * mcat([proposedA, OMPCSEMI, sc]))
# 
#         MC(((tr - 1) * r) + (sc)).lvalue = MC(((tr - 1) * r) + (sc)) + W(((tr - 1) * r) + (sc))
# 
#     MC = MC / Niter
#     ridgetable = 1 - MC
