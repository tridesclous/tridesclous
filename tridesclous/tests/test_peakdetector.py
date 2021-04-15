from tridesclous import get_dataset
from tridesclous.peakdetector import get_peak_detector_class

import time
import itertools

import scipy.signal
import numpy as np
import sklearn.metrics.pairwise

from matplotlib import pyplot

from tridesclous.tests.test_signalpreprocessor import offline_signal_preprocessor

from tridesclous.peakdetector import make_sum_rectified, detect_peaks_in_rectified, get_mask_spatiotemporal_peaks
from tridesclous.peakdetector import HAVE_PYOPENCL

import matplotlib.pyplot as plt





def get_normed_sigs(chunksize=None):
    # get sigs
    sigs, sample_rate = get_dataset(name='olfactory_bulb')
    #~ sigs = np.tile(sigs, (1, 20)) #for testing large channels num
    
    if sigs.shape[0] % chunksize >0:
        sigs = sigs[:-(sigs.shape[0] % chunksize), :]
    
    nb_channel = sigs.shape[1]
    #~ print('nb_channel', nb_channel)
    
    geometry = np.zeros((nb_channel, 2))
    geometry[:, 0] = np.arange(nb_channel) * 50 # um spacing

    
    
    
    # normalize sigs
    highpass_freq = 300.
    preprocess_params = dict(
                highpass_freq=highpass_freq,
                common_ref_removal=True,
                backward_chunksize=chunksize+chunksize//4,
                output_dtype='float32')
    normed_sigs = offline_signal_preprocessor(sigs, sample_rate, **preprocess_params)    
    
    return sigs, sample_rate, normed_sigs, geometry




def offline_peak_detect_global(normed_sigs, sample_rate, geometry, 
                peak_sign='-',relative_threshold = 5,  peak_span_ms=0.5, smooth_radius_um=None):
    
    
    n_span = int(sample_rate * peak_span_ms / 1000.)//2
    
    if smooth_radius_um is None:
        spatial_matrix = None
    else:
        d = sklearn.metrics.pairwise.euclidean_distances(geometry)
        spatial_matrix = np.exp(-d/smooth_radius_um)
        spatial_matrix[spatial_matrix<0.01] = 0.
    
    sum_rectified = make_sum_rectified(normed_sigs, relative_threshold, peak_sign, spatial_matrix)
    mask_peaks = detect_peaks_in_rectified(sum_rectified, n_span, relative_threshold, peak_sign)
    ind_peaks,  = np.nonzero(mask_peaks)
    ind_peaks += n_span
    
    return ind_peaks, sum_rectified



def offline_peak_detect_geometrical(normed_sigs, sample_rate, geometry, 
                peak_sign='-',relative_threshold = 5,  peak_span_ms=0.5, 
                adjacency_radius_um=None, smooth_radius_um=None):
    
    assert smooth_radius_um is None
    assert adjacency_radius_um is not None
    
    nb_channel = normed_sigs.shape[1]
    n_span = int(sample_rate * peak_span_ms / 1000.)//2
    
    d = sklearn.metrics.pairwise.euclidean_distances(geometry)
    neighbour_mask = d<=adjacency_radius_um
    nb_neighbour_per_channel = np.sum(neighbour_mask, axis=0)
    nb_max_neighbour = np.max(nb_neighbour_per_channel)
    
    nb_max_neighbour = nb_max_neighbour
    neighbours = np.zeros((nb_channel, nb_max_neighbour), dtype='int32')
    neighbours[:] = -1
    for c in range(nb_channel):
        neighb, = np.nonzero(neighbour_mask[c, :])
        neighbours[c, :neighb.size] = neighb
    
    peak_mask = get_mask_spatiotemporal_peaks(normed_sigs, n_span, relative_threshold, peak_sign, neighbours)
    peaks, chan_inds = np.nonzero(peak_mask)
    
    return peaks



def test_compare_offline_online_engines():
    #~ HAVE_PYOPENCL = True

    engine_names = [
        ('global', 'numpy'),
        ('geometrical', 'numpy'),
        ('geometrical', 'numba'),
    ]
    
    if HAVE_PYOPENCL:
        #~ engine_names += [('global', 'opencl'),
                            #~ ('geometrical', 'opencl')]
        engine_names += [('geometrical', 'opencl')]
                        

    chunksize=1024
    sigs, sample_rate, normed_sigs, geometry = get_normed_sigs(chunksize=chunksize)

    #params
    peak_sign = '-'
    relative_threshold = 8
    peak_span_ms = 0.9
    smooth_radius_um = None
    adjacency_radius_um = 200.

    
    nb_channel = sigs.shape[1]
    
    #~ print('n_span', n_span)
    nloop = sigs.shape[0]//chunksize
    
    
    print('sig duration', sigs.shape[0]/sample_rate)
    
    offline_peaks = {}
    t1 = time.perf_counter()
    peaks, rectified_sum = offline_peak_detect_global(sigs, sample_rate, geometry, 
                                    peak_sign=peak_sign, relative_threshold=relative_threshold, peak_span_ms=peak_span_ms, 
                                    smooth_radius_um=smooth_radius_um)
    t2 = time.perf_counter()
    print('offline global', 'process time', t2-t1)
    offline_peaks['global', 'numpy'] = peaks
    offline_peaks['global', 'opencl'] = peaks

    t1 = time.perf_counter()
    peaks = offline_peak_detect_geometrical(sigs, sample_rate, geometry, 
                                    peak_sign=peak_sign, relative_threshold=relative_threshold, peak_span_ms=peak_span_ms, 
                                    smooth_radius_um=smooth_radius_um, adjacency_radius_um=adjacency_radius_um)
    t2 = time.perf_counter()
    print('offline geometrical', 'process time', t2-t1)

    offline_peaks['geometrical', 'numpy'] = peaks
    offline_peaks['geometrical', 'numba'] = peaks
    offline_peaks['geometrical', 'opencl'] = peaks

    online_peaks = {}
    for method, engine in engine_names:
        print(engine)
        EngineClass = get_peak_detector_class(method, engine)
        
        #~ buffer_size = chunksize*4
        peakdetector = EngineClass(sample_rate, nb_channel, chunksize, 'float32', geometry)
        
        peakdetector.change_params(peak_sign=peak_sign, relative_threshold=relative_threshold,
                        peak_span_ms=peak_span_ms, smooth_radius_um=smooth_radius_um,
                        adjacency_radius_um=adjacency_radius_um)
        
        all_online_peaks = []
        t1 = time.perf_counter()
        for i in range(nloop):
            #~ print(i)
            pos = (i+1)*chunksize
            chunk = sigs[pos-chunksize:pos,:]
            time_ind_peaks, chan_peak_index, peak_val_peaks = peakdetector.process_buffer_stream(pos, chunk)
            #~ print(n_peaks)
            if time_ind_peaks is not None:
                #~ all_online_peaks.append(chunk_peaks['index'])
                all_online_peaks.append(time_ind_peaks)
        online_peaks[method, engine] = np.concatenate(all_online_peaks)
        t2 = time.perf_counter()
        print(engine, 'process time', t2-t1, 'size', online_peaks[method, engine].size)
    
    # remove peaks on border for comparison
    for method, engine in engine_names:
        peaks = online_peaks[method, engine]
        peaks = peaks[(peaks>chunksize) & (peaks<sigs.shape[0]-chunksize)]
        online_peaks[method, engine] = peaks

        peaks = offline_peaks[method, engine]
        peaks = peaks[(peaks>chunksize) & (peaks<sigs.shape[0]-chunksize)]
        offline_peaks[method, engine] = peaks
        
    # compare
    for method, engine in engine_names:
        print('compare', method, engine)
        onlinepeaks = online_peaks[method, engine]
        offlinepeaks = offline_peaks[method, engine]
        print(onlinepeaks.size, offlinepeaks.size)
        # TODO
        #~ assert offlinepeaks.size==onlinepeaks.size, '{} nb_peak {} instead {}'.format(engine,  offlinepeaks.size, onlinepeaks.size)
        #~ assert np.array_equal(offlinepeaks, onlinepeaks)
    

def test_detect_geometrical_peaks():
    chunksize=1024
    sigs, sample_rate, normed_sigs, geometry = get_normed_sigs(chunksize=chunksize)
    nb_channel = sigs.shape[1]
    
    n_span = 4
    thresh = 5
    peak_sign = '-'

    d = sklearn.metrics.pairwise.euclidean_distances(geometry)
    nb_neighbour = 4
    neighbours = np.zeros((nb_channel, nb_neighbour+1), dtype='int64')
    for c in range(nb_channel):
        nearest = np.argsort(d[c, :])
        #~ print(c, nearest)
        neighbours[c, :] = nearest[:nb_neighbour+1] # include itself
    #~ print(neighbours)
    
    mask = get_mask_spatiotemporal_peaks(normed_sigs, n_span, thresh, peak_sign, neighbours)
    
    peak_inds, chan_inds = np.nonzero(mask)
    peak_inds += n_span
    
    print(peak_inds.size)
    
    #~ fig, ax = plt.subplots()
    #~ plot_sigs = normed_sigs.copy()
    #~ for c in range(nb_channel):
        #~ plot_sigs[:, c] += c*30
    #~ ax.plot(plot_sigs, color='k')
    #~ ampl = plot_sigs[peak_inds, chan_inds]
    #~ ax.scatter(peak_inds, ampl, color='r')
    #~ plt.show()
    
    
    # test two way
    mask_neg = get_mask_spatiotemporal_peaks(normed_sigs, n_span, thresh, '-', neighbours)
    mask_pos = get_mask_spatiotemporal_peaks(-normed_sigs, n_span, thresh, '+', neighbours)
    assert np.array_equal(mask_neg, mask_pos)
    
    
    
    #~ print(peak_inds)
    #~ print(chan_inds)
    



def benchmark_speed():
    chunksize=1024
    
    #~ chunksize=1025
    #~ chunksize= 1024 + 256
    #~ chunksize=2048
    #~ chunksize = 1024 * 10
    #~ chunksize=950
    
    sigs, sample_rate, normed_sigs, geometry = get_normed_sigs(chunksize=chunksize)
    
    #~ sigs = np
    
    #***for testing large channels num***
    sigs = np.tile(sigs, (1, 20))
    normed_sigs = np.tile(normed_sigs, (1, 20))
    geometry = np.zeros((sigs.shape[1], 2), dtype='float64')
    geometry[:, 0] = np.arange(sigs.shape[1]) * 50.
    #***
    
    
    nb_channel = sigs.shape[1]
    print('nb_channel', nb_channel)

    engine_names = [
        #~ ('global', 'numpy'),
        #~ ('geometrical', 'numpy'),
        ('geometrical', 'numba'),
    ]
    if HAVE_PYOPENCL:
        engine_names += [
            #~ ('global', 'opencl'),
            ('geometrical', 'opencl'),
        ]

    args = (sample_rate, nb_channel, chunksize, 'float32', geometry)
    params = dict(peak_span_ms = 0.9,
                    relative_threshold = 5,
                    peak_sign = '-')
    
    online_peaks = {}
    
    

    for method, engine in engine_names:
        peakdetector = get_peak_detector_class(method, engine)(*args)
        peakdetector.change_params(**params)
        
        #~ print(peakdetector.n_span, peakdetector.dtype)
            
        nloop = normed_sigs.shape[0]//chunksize
        peak_inds = []
        peak_chans = []
        t1 = time.perf_counter()
        for i in range(nloop):
            pos = (i+1)*chunksize
            chunk = normed_sigs[pos-chunksize:pos,:]
            time_ind_peaks, chan_peak_index, peak_val_peaks = peakdetector.process_buffer_stream(pos, chunk)
            if time_ind_peaks is not None:
                peak_inds.append(time_ind_peaks)
                
                if chan_peak_index is not None:
                    peak_chans.append(chan_peak_index)
        t2 = time.perf_counter()
        
        peak_inds = np.concatenate(peak_inds)
        if len(peak_chans) > 0:
            peak_chans = np.concatenate(peak_chans)
        else:
            peak_chans = np.argmin(normed_sigs[peak_inds, :], axis=1)
            
        online_peaks[method, engine] = peak_inds
        
        print(method, engine, ':' , peak_inds.size)
        print(method, engine, 'process time', t2-t1) 
        
            
        
        #~ fig, ax = plt.subplots()
        #~ plot_sigs = normed_sigs.copy()
        #~ for c in range(nb_channel):
            #~ plot_sigs[:, c] += c*30
        #~ ax.plot(plot_sigs, color='k')
        #~ ampl = plot_sigs[peak_inds, peak_chans]
        #~ ax.scatter(peak_inds, ampl, color='r')
        #~ plt.show()
        


def test_peak_sign_symetry():
    chunksize=1024
    
    raw_sigs, sample_rate, normed_sigs, geometry = get_normed_sigs(chunksize=chunksize)
    nb_channel = normed_sigs.shape[1]
    #~ print('nb_channel', nb_channel)

    args = (sample_rate, nb_channel, chunksize, 'float32', geometry)
    params = dict(peak_span_ms = 0.9,
                    relative_threshold = 5)
    
    engine_names = [
        ('global', 'numpy'),
        ('geometrical', 'numpy'),
        ('geometrical', 'numba'),
    ]
    if HAVE_PYOPENCL:
        engine_names += [
            ('global', 'opencl'),
            ('geometrical', 'opencl'),
        ]
    online_peaks = {}
    for method, engine in engine_names:
        peakdetector = get_peak_detector_class(method, engine)(*args)
        
        
        for peak_sign in ['-', '+']:
        

            if peak_sign=='-':
                sigs = normed_sigs
            elif peak_sign=='+':
                sigs = -normed_sigs            
        
            peakdetector.change_params(peak_sign=peak_sign, **params)
            
            nloop = normed_sigs.shape[0]//chunksize
            peaks = []
            t1 = time.perf_counter()
            for i in range(nloop):
                #~ print(i)
                pos = (i+1)*chunksize
                chunk = sigs[pos-chunksize:pos,:]
                #~ print(chunk.shape)
                time_ind_peaks, chan_peak_index, peak_val_peaks = peakdetector.process_buffer_stream(pos, chunk)
                #~ print(n_peaks)
                #~ print(chunk_peaks)
                if time_ind_peaks is not None:
                    #~ all_online_peaks.append(chunk_peaks['index'])
                    peaks.append(time_ind_peaks)
            peak_inds = np.concatenate(peaks)
            online_peaks[method, engine, peak_sign] = peak_inds
            t2 = time.perf_counter()
            print(method, engine, 'peak_sign', peak_sign,':' , peak_inds.size, 'unique peak size', np.unique(peak_inds).size)
            #~ print(name, 'process time', t2-t1) 

        assert np.array_equal(online_peaks[method, engine, '-'], online_peaks[method, engine, '+'])
    
    if HAVE_PYOPENCL:
        assert np.array_equal(online_peaks['global', 'numpy', '-'], online_peaks['global', 'opencl', '-'])
        assert np.array_equal(online_peaks['geometrical', 'numpy', '-'], online_peaks['geometrical', 'numba', '-'])
        
        
        # TODO this should be totally equal
        assert np.array_equal(online_peaks['geometrical', 'numpy', '-'], online_peaks['geometrical', 'opencl', '-'])
        assert np.array_equal(online_peaks['geometrical', 'numba', '-'], online_peaks['geometrical', 'opencl', '-'])

    

    
if __name__ == '__main__':
    test_compare_offline_online_engines()
    
    #~ test_detect_geometrical_peaks()
    
    
    #~ benchmark_speed()
    
    #~ test_peak_sign_symetry()
    