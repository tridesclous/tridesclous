from tridesclous import get_dataset
from tridesclous.peakdetector import peakdetector_engines

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


def offline_peak_detect(normed_sigs, sample_rate, geometry, 
                peak_sign='-',relative_threshold = 5,  peak_span_ms=0.5, adjacency_radius_um=None):
    
    
    n_span = int(sample_rate * peak_span_ms / 1000.)//2
    
    if adjacency_radius_um is None:
        spatial_matrix = None
    else:
        d = sklearn.metrics.pairwise.euclidean_distances(geometry)
        spatial_matrix = np.exp(-d/adjacency_radius_um)
        spatial_matrix[spatial_matrix<0.01] = 0.
    
    sum_rectified = make_sum_rectified(normed_sigs, relative_threshold, peak_sign, spatial_matrix)
    mask_peaks = detect_peaks_in_rectified(sum_rectified, n_span, relative_threshold, peak_sign)
    ind_peaks,  = np.nonzero(mask_peaks)
    ind_peaks += n_span
    
    return ind_peaks, sum_rectified



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

def test_compare_offline_online_engines():
    #~ HAVE_PYOPENCL = True
    #~ if HAVE_PYOPENCL:
        #~ engines = ['numpy', 'opencl']
        #~ engines = [ 'opencl']
        #~ engines = ['numpy']
    #~ else:
        #~ engines = ['numpy']
        
    engines = ['numpy']

    #params
    peak_sign = '-'
    relative_threshold = 8
    peak_span_ms = 0.9
    adjacency_radius_um = None

    chunksize=1024
    sigs, sample_rate, normed_sigs, geometry = get_normed_sigs(chunksize=chunksize)
    
    nb_channel = sigs.shape[1]
    
    #~ print('n_span', n_span)
    nloop = sigs.shape[0]//chunksize
    
    
    print('sig duration', sigs.shape[0]/sample_rate)
    
    
    #~ for peak_sign in ['-', '+', ]:
    for peak_sign, adjacency_radius_um in itertools.product(['-', '+'], [None, 100]):
    #~ for peak_sign in ['+', ]:
    #~ for peak_sign in ['-', ]:
        print()
        print('peak_sign', peak_sign, 'adjacency_radius_um', adjacency_radius_um)
        if peak_sign=='-':
            sigs = normed_sigs
        elif peak_sign=='+':
            sigs = -normed_sigs
        
        #~ print(sigs.shape)
        #~ print('nloop', nloop)
        
        
        t1 = time.perf_counter()
        offline_peaks, rectified_sum = offline_peak_detect(sigs, sample_rate, geometry, 
                                        peak_sign=peak_sign, relative_threshold=relative_threshold, peak_span_ms=peak_span_ms, 
                                        adjacency_radius_um=adjacency_radius_um)
        t2 = time.perf_counter()
        print('offline', 'process time', t2-t1)
        #~ print(offline_peaks)
        
        online_peaks = {}
        for engine in engines:
            print(engine)
            EngineClass = peakdetector_engines[engine]
            #~ buffer_size = chunksize*4
            peakdetector_engine = EngineClass(sample_rate, nb_channel, chunksize, 'float32', geometry)
            
            peakdetector_engine.change_params(peak_sign=peak_sign, relative_threshold=relative_threshold,
                            peak_span_ms=peak_span_ms, adjacency_radius_um=adjacency_radius_um)
            
            all_online_peaks = []
            t1 = time.perf_counter()
            for i in range(nloop):
                #~ print(i)
                pos = (i+1)*chunksize
                chunk = sigs[pos-chunksize:pos,:]
                n_peaks, chunk_peaks = peakdetector_engine.process_data(pos, chunk)
                #~ print(n_peaks)
                if chunk_peaks is not None:
                    #~ all_online_peaks.append(chunk_peaks['index'])
                    all_online_peaks.append(chunk_peaks)
            online_peaks[engine] = np.concatenate(all_online_peaks)
            t2 = time.perf_counter()
            print(engine, 'process time', t2-t1)
        
        # remove peaks on border for comparison
        offline_peaks = offline_peaks[(offline_peaks>chunksize) & (offline_peaks<sigs.shape[0]-chunksize)]
        for engine in engines:
            onlinepeaks = online_peaks[engine]
            onlinepeaks = onlinepeaks[(onlinepeaks>chunksize) & (onlinepeaks<sigs.shape[0]-chunksize)]
            online_peaks[engine] = onlinepeaks

        # compare
        for engine in engines:
            onlinepeaks = online_peaks[engine]
            assert offline_peaks.size==onlinepeaks.size, '{} nb_peak{} instead {}'.format(engine,  offline_peaks.size, onlinepeaks.size)
            assert np.array_equal(offline_peaks, onlinepeaks)
    
        # plot
        #~ fig, axs = pyplot.subplots(nrows=nb_channel, sharex=True)
        #~ for i in range(nb_channel):
            #~ axs[i].plot(sigs[:, i])
            #~ axs[i].plot(offline_peaks, sigs[offline_peaks, i], ls = 'None', marker = 'o', color='g', markersize=12)
            #~ for engine in engines:
                #~ onlinepeaks = online_peaks[engine]
                #~ axs[i].plot(onlinepeaks, sigs[onlinepeaks, i], ls = 'None', marker = 'o', color='r', markersize=6)
        
        #~ fig, ax = pyplot.subplots()
        #~ ax.plot(rectified_sum)
        #~ ax.plot(offline_peaks, rectified_sum[offline_peaks], ls = 'None', marker = 'o', color='g', markersize=12)
        #~ for engine in engines:
            #~ onlinepeaks = online_peaks[engine]
            #~ ax.plot(onlinepeaks, rectified_sum[onlinepeaks], ls = 'None', marker = 'o', color='r', markersize=6)
        
        #~ for i in range(nloop):
            #~ ax.axvline(i*chunksize, color='k', alpha=0.4)
        
        #~ pyplot.show()



def test_detect_spatiotemporal_peaks():
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
    
    fig, ax = plt.subplots()
    plot_sigs = normed_sigs.copy()
    for c in range(nb_channel):
        plot_sigs[:, c] += c*30
    ax.plot(plot_sigs, color='k')
    ampl = plot_sigs[peak_inds, chan_inds]
    ax.scatter(peak_inds, ampl, color='r')
    plt.show()
    
    
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

    args = (sample_rate, nb_channel, chunksize, 'float32', geometry)
    peak_detectors = {
        #~ 'numpy' : peakdetector_engines['numpy'](*args),
        #~ 'opencl' : peakdetector_engines['opencl'](*args),
        #~ 'spatiotemporal' : peakdetector_engines['spatiotemporal'](*args),
        'spatiotemporal_opencl' : peakdetector_engines['spatiotemporal_opencl'](*args),
    }
    
    params = dict(peak_span_ms = 0.9,
                    relative_threshold = 5,
                    peak_sign = '-')
    
    online_peaks = {}
    
    
    for name, peakdetector in peak_detectors.items():
        
        
        peakdetector.change_params(**params)
        #~ print(peakdetector.n_span, peakdetector.dtype)
            
        nloop = normed_sigs.shape[0]//chunksize
        peak_inds = []
        peak_chans = []
        t1 = time.perf_counter()
        for i in range(nloop):
            pos = (i+1)*chunksize
            chunk = normed_sigs[pos-chunksize:pos,:]
            time_ind_peaks, chan_peak_index = peakdetector.process_data(pos, chunk)
            if time_ind_peaks is not None:
                peak_inds.append(time_ind_peaks)
                
                if chan_peak_index is not None:
                    peak_chans.append(chan_peak_index)
        t2 = time.perf_counter()
        
        peak_inds = np.concatenate(peak_inds)
        
        online_peaks[name] = peak_inds
        
        print(name, ':' , peak_inds.size)
        print(name, 'process time', t2-t1) 
        

        peak_chans = np.concatenate(peak_chans)
        fig, ax = plt.subplots()
        plot_sigs = normed_sigs.copy()
        for c in range(nb_channel):
            plot_sigs[:, c] += c*30
        ax.plot(plot_sigs, color='k')
        ##ind_min = np.argmin(normed_sigs[peak_inds, :], axis=1)
        ampl = plot_sigs[peak_inds, peak_chans]
        ax.scatter(peak_inds, ampl, color='r')
        plt.show()        
        


def test_peak_sign_symetry():
    chunksize=1024
    
    raw_sigs, sample_rate, normed_sigs, geometry = get_normed_sigs(chunksize=chunksize)
    nb_channel = normed_sigs.shape[1]
    #~ print('nb_channel', nb_channel)

    args = (sample_rate, nb_channel, chunksize, 'float32', geometry)
    peak_detectors = {
        'numpy' : peakdetector_engines['numpy'](*args),
        'opencl' : peakdetector_engines['opencl'](*args),
        'spatiotemporal' : peakdetector_engines['spatiotemporal'](*args),
        'spatiotemporal_opencl' : peakdetector_engines['spatiotemporal_opencl'](*args),
    }
    
    params = dict(peak_span_ms = 0.9,
                    relative_threshold = 5)
                    #~ peak_sign = '-')
    
    
    args = (sample_rate, nb_channel, chunksize, 'float32', geometry)
    
    engine_names = [
        'numpy',
        'opencl',
        'spatiotemporal',
        'spatiotemporal_opencl',
    ]
    
    params = dict(peak_span_ms = 0.9, relative_threshold = 5)

    online_peaks = {}
    for name in engine_names:
        peakdetector = peakdetector_engines[name](*args)
        
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
                time_ind_peaks, chan_peak_index = peakdetector.process_data(pos, chunk)
                #~ print(n_peaks)
                #~ print(chunk_peaks)
                if time_ind_peaks is not None:
                    #~ all_online_peaks.append(chunk_peaks['index'])
                    peaks.append(time_ind_peaks)
            peak_inds = np.concatenate(peaks)
            online_peaks[name, peak_sign] = peak_inds
            t2 = time.perf_counter()
            print(name, 'peak_sign', peak_sign,':' , peak_inds.size, 'unique peak size', np.unique(peak_inds).size)
            #~ print(name, 'process time', t2-t1) 

        assert np.array_equal(online_peaks[name, '-'], online_peaks[name, '+'])
    
    assert np.array_equal(online_peaks['numpy', '-'], online_peaks['opencl', '-'])
    assert np.array_equal(online_peaks['spatiotemporal', '-'], online_peaks['spatiotemporal_opencl', '-'])
    
    
        
    

    
if __name__ == '__main__':
    #~ test_compare_offline_online_engines()
    
    #~ test_detect_spatiotemporal_peaks()
    
    
    benchmark_speed()
    
    #~ test_peak_sign_symetry()
    