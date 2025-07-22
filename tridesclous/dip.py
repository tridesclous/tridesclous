"""
Module for computing The Hartigans' dip statistic

This module is more or less a copy/paste from
https://github.com/tatome/dip_test/blob/master/dip.py
Johannes Bauer

and 
https://github.com/BenjaminDoran/unidip
BenjaminDoran


    
The dip statistic measures unimodality of a sample from a random process.
See: 
Hartigan, J. A.; Hartigan, P. M. The Dip Test of Unimodality. The Annals 
of Statistics 13 (1985), no. 1, 70--84. doi:10.1214/aos/1176346577. 
http://projecteuclid.org/euclid.aos/1176346577.
"""

import collections
import numpy as np

def _gcm_(cdf, idxs):
    work_cdf = cdf
    work_idxs = idxs
    gcm = [work_cdf[0]]
    touchpoints = [0]
    while len(work_cdf) > 1:
        distances = work_idxs[1:] - work_idxs[0]
        slopes = (work_cdf[1:] - work_cdf[0]) / distances
        minslope = slopes.min()
        minslope_idx = np.where(slopes == minslope)[0][0] + 1
        gcm.extend(work_cdf[0] + distances[:minslope_idx] * minslope)
        touchpoints.append(touchpoints[-1] + minslope_idx)
        work_cdf = work_cdf[minslope_idx:]
        work_idxs = work_idxs[minslope_idx:]
    return np.array(np.array(gcm)), np.array(touchpoints)

def _lcm_(cdf, idxs):
    g, t = _gcm_(1-cdf[::-1], idxs.max() - idxs[::-1])
    return 1-g[::-1], len(cdf) - 1 - t[::-1]

def _touch_diffs_(part1, part2, touchpoints):
    diff = np.abs((part2[touchpoints] - part1[touchpoints]))
    return diff.max(), diff

def diptest(dat, is_hist=False, numt=1000):
    """ diptest with pval """
    # sample dip
    d, (_, idxs, left, _, right, _) = dip_fn(dat, is_hist)

    # simulate from null uniform
    unifs = np.random.uniform(size=numt * idxs.shape[0])\
                     .reshape([numt, idxs.shape[0]])
    unif_dips = np.apply_along_axis(dip_fn, 1, unifs, is_hist, True)

    # count dips greater or equal to d, add 1/1 to prevent a pvalue of 0
    pval = None \
      if unif_dips.sum() == 0 else \
        (np.less(d, unif_dips).sum() + 1) / (np.float64(numt) + 1)
    
    return pval
    #~ return (d, pval, # dip, pvalue
            #~ (len(left)-1, len(idxs)-len(right)) # indices
           #~ )

def dip_fn(dat, is_hist=False, just_dip=False):
    """
        Compute the Hartigans' dip statistic either for a histogram of
        samples (with equidistant bins) or for a set of samples.
    """
    if is_hist:
        histogram = dat
        idxs = np.arange(len(histogram))
    else:
        counts = collections.Counter(dat)
        idxs = np.sort(list(counts.keys()), axis=0)
        histogram = np.array([counts[i] for i in idxs])

    # check for case 1<N<4 or all identical values
    if len(idxs) <= 4 or idxs[0] == idxs[-1]:
        left = []
        right = [1]
        d = 0.0
        return d if just_dip else (d, (None, idxs, left, None, right, None))

    cdf = np.cumsum(histogram, dtype=float)
    cdf /= cdf[-1]

    work_idxs = idxs
    work_histogram = np.asarray(histogram, dtype=float) / np.sum(histogram)
    work_cdf = cdf

    D = 0
    left = [0]
    right = [1]

    while True:
        left_part, left_touchpoints = _gcm_(work_cdf-work_histogram, work_idxs)
        right_part, right_touchpoints = _lcm_(work_cdf, work_idxs)

        d_left, left_diffs = _touch_diffs_(left_part,
                                           right_part, left_touchpoints)
        d_right, right_diffs = _touch_diffs_(left_part,
                                             right_part, right_touchpoints)

        if d_right > d_left:
            xr = right_touchpoints[d_right == right_diffs][-1]
            xl = left_touchpoints[left_touchpoints <= xr][-1]
            d = d_right
        else:
            xl = left_touchpoints[d_left == left_diffs][0]
            xr = right_touchpoints[right_touchpoints >= xl][0]
            d = d_left

        left_diff = np.abs(left_part[:xl+1] - work_cdf[:xl+1]).max()
        right_diff = np.abs(right_part[xr:]
                            - work_cdf[xr:]
                            + work_histogram[xr:]).max()

        if d <= D or xr == 0 or xl == len(work_cdf):
            the_dip = max(np.abs(cdf[:len(left)] - left).max(),
                          np.abs(cdf[-len(right)-1:-1] - right).max())
            if just_dip:
                return the_dip/2
            else:
                return the_dip/2, (cdf, idxs, left, left_part, right, right_part)
        else:
            D = max(D, left_diff, right_diff)

        work_cdf = work_cdf[xl:xr+1]
        work_idxs = work_idxs[xl:xr+1]
        work_histogram = work_histogram[xl:xr+1]

        left[len(left):] = left_part[1:xl+1]
        right[:0] = right_part[xr:-1]

