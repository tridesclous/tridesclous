import numpy as np
import scipy.signal
import time


from tridesclous import get_dataset
from tridesclous.signalpreprocessor import SignalPreprocessor_Numpy
from tridesclous.peakdetector import PeakDetectorEngine_Numpy
from tridesclous.waveformextractor import OnlineWaveformExtractor, cut_full



from tridesclous.tests.test_signalpreprocessor import offline_signal_preprocessor
from tridesclous.tests.test_peakdetector import offline_peak_detect


from matplotlib import pyplot




def test_compare_offline_online_engines():

    sigs, sample_rate = get_dataset()
    #~ sigs = sigs[:, [0]]
    nb_channel = sigs.shape[1]
    print('nb_channel', nb_channel)

    #params
    chunksize = 1024
    nloop = sigs.shape[0]//chunksize
    sigs = sigs[:chunksize*nloop]
    highpass_freq = 300.
    preprocess_params = dict(
                highpass_freq=highpass_freq,
                common_ref_removal=True,
                backward_chunksize=chunksize+chunksize//4,
                output_dtype='float32')
    
    peak_params = dict(peak_sign='-',
                                    relative_threshold=8,
                                    peak_span = 0.0009)
    
    waveforms_params = dict(n_left=-20, n_right=30)
    
    n_left = -20
    n_right = 30
    
    t1 = time.perf_counter()
    offline_sig = offline_signal_preprocessor(sigs, sample_rate, **preprocess_params)
    offline_peaks, rectified_sum = offline_peak_detect(offline_sig, sample_rate, **peak_params)
    keep = (offline_peaks>chunksize) & (offline_peaks<sigs.shape[0]-chunksize)
    offline_peaks = offline_peaks[keep]
    offline_waveforms = cut_full(offline_sig, offline_peaks+n_left, n_right-n_left)
    print(offline_waveforms.shape)
    t2 = time.perf_counter()
    print('offline', 'process time', t2-t1)
    
    # precompute medians and mads
    params2 = dict(preprocess_params)
    params2['normalize'] = False
    sigs_for_noise = offline_signal_preprocessor(sigs, sample_rate, **params2)
    medians = np.median(sigs_for_noise, axis=0)
    mads = np.median(np.abs(sigs_for_noise-medians),axis=0)*1.4826
    preprocess_params['signals_medians'] = medians
    preprocess_params['signals_mads'] = mads
    #
    
    
    signalpreprocessor = SignalPreprocessor_Numpy(sample_rate, nb_channel, chunksize, sigs.dtype)
    signalpreprocessor.change_params(**preprocess_params)
    
    peakdetector = PeakDetectorEngine_Numpy(sample_rate, nb_channel, chunksize, 'float32')
    peakdetector.change_params(**peak_params)
            
    waveformextractor = OnlineWaveformExtractor(nb_channel, chunksize)
    waveformextractor.change_params(**waveforms_params)
    

    all_online_peak = []
    all_online_waveforms = []
    
    t1 = time.perf_counter()
    for i in range(nloop):
        #~ print()
        pos = (i+1)*chunksize
        #~ print('loop', i, 'pos', pos-chunksize, pos)
        
        chunk = sigs[pos-chunksize:pos,:]
        
        pos2, preprocessed_chunk = signalpreprocessor.process_data(pos, chunk)
        if preprocessed_chunk is  None:
            continue
        
        #~ print('pos2', pos)
        
        n_peaks, chunk_peaks = peakdetector.process_data(pos2, preprocessed_chunk)
        if chunk_peaks is  None:
            continue
        
        for peak_pos, chunk_waveforms in waveformextractor.new_peaks(pos2, preprocessed_chunk, chunk_peaks):
            #~ print(peak_pos, chunk_waveforms.shape)
            all_online_peak.append(peak_pos)
            all_online_waveforms.append(chunk_waveforms)
            
    t2 = time.perf_counter()
    print('online process time', t2-t1)
    
    online_peaks = np.concatenate(all_online_peak, axis=0)
    online_waveforms = np.concatenate(all_online_waveforms, axis=0)
    
    keep = (online_peaks>chunksize) & (online_peaks<sigs.shape[0]-chunksize)
    online_peaks = online_peaks[keep]
    online_waveforms = online_waveforms[keep]
    
    assert np.array_equal(offline_peaks, online_peaks)
    

    residual = np.abs((online_waveforms.astype('float64')-offline_waveforms.astype('float64'))/np.mean(np.abs(offline_waveforms.astype('float64'))))

    print(np.max(residual))
    #~ print(np.mean(np.abs(offline_sig.astype('float64'))))
    assert np.max(residual)<5e-5, 'online differt from offline'

    
    
    ind_error_max = np.argmax(np.max(residual.reshape(residual.shape[0], -1), axis=1))
    #~ print(ind_error_max)
    
    offline_wf = offline_waveforms[ind_error_max, : , :]
    online_wf = online_waveforms[ind_error_max, : , :]
    #~ print(online_wf.shape)
    
    fig, ax = pyplot.subplots()
    ax.plot(offline_wf.flatten(), color='g')
    ax.plot(online_wf.flatten(), color='r', ls='--')
    
    
    fig, ax = pyplot.subplots()
    wf2 = offline_waveforms.reshape(offline_waveforms.shape[0], -1)
    ax.plot(np.median(wf2, axis=0), color='g')
    wf3 = online_waveforms.reshape(offline_waveforms.shape[0], -1)
    ax.plot(np.median(wf3, axis=0), color='r', ls='--')
    
    
    pyplot.show()
    




if __name__ == '__main__':
    test_compare_offline_online_engines()
    


