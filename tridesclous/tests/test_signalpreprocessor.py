from tridesclous import get_dataset
from tridesclous.signalpreprocessor import signalpreprocessor_engines

import time

import scipy.signal
import numpy as np

from matplotlib import pyplot



def offline_signal_preprocessor(sigs, sample_rate, highpass_freq=300.,output_dtype='float32', **unused):
    #cast
    sigs = sigs.astype(output_dtype)
    
    #filter
    b, a = scipy.signal.iirfilter(5, highpass_freq/sample_rate*2, analog=False,
                                    btype = 'highpass', ftype = 'butter', output = 'ba')
    filtered_sigs = scipy.signal.filtfilt(b, a, sigs, axis=0)
    
    # normalize
    med = np.median(filtered_sigs, axis=0)
    mad = np.median(np.abs(filtered_sigs-med),axis=0)*1.4826
    normed_sigs = (filtered_sigs - med)/mad
    
    return normed_sigs.astype(output_dtype)
    




def test_compare_offline_online_engines():
    HAVE_PYOPENCL = True
    if HAVE_PYOPENCL:
        #~ engines = ['signalpreprocessor_numpy', 'signalpreprocessor_opencl']
        engines = ['signalpreprocessor_numpy']
        
    else:
        engines = ['signalpreprocessor_numpy']

    
    sigs, sample_rate = get_dataset()
    #~ sigs = sigs[:, [0]]
    nb_channel = sigs.shape[1]
    print('nb_channel', nb_channel)
    
    #params
    chunksize = 1024
    nloop = sigs.shape[0]//chunksize
    sigs = sigs[:chunksize*nloop]
    
    print('sig duration', sigs.shape[0]/sample_rate)
    
    highpass_freq = 300.
    params = {'highpass_freq': highpass_freq, 'output_dtype': 'float32', 
                'backward_chunksize':chunksize+chunksize//4}
    
    t1 = time.perf_counter()
    offline_sig = offline_signal_preprocessor(sigs, sample_rate, **params)
    t2 = time.perf_counter()
    print('offline', 'process time', t2-t1)
    
    # precompute medians and mads
    b, a = scipy.signal.iirfilter(5, highpass_freq/sample_rate*2, analog=False,
                                    btype = 'highpass', ftype = 'butter', output = 'ba')
    filtered_sigs = scipy.signal.filtfilt(b, a, sigs, axis=0)
    medians = np.median(filtered_sigs, axis=0)
    mads = np.median(np.abs(filtered_sigs-medians),axis=0)*1.4826
    params['medians'] = medians
    params['mads'] = mads
    
    
    online_sigs = {}
    for engine in engines:
        print(engine)
        SignalPreprocessorClass = signalpreprocessor_engines[engine]
        signalpreprocessor = SignalPreprocessorClass(sample_rate, nb_channel, chunksize, sigs.dtype)
        signalpreprocessor.change_params(**params)
        
        all_online_sigs = []
        t1 = time.perf_counter()
        for i in range(nloop):
            #~ print(i)
            pos = (i+1)*chunksize
            chunk = sigs[pos-chunksize:pos,:]
            pos2, preprocessed_chunk = signalpreprocessor.process_data(pos, chunk)
            if preprocessed_chunk is not None:
                #~ print(preprocessed_chunk)
                all_online_sigs.append(preprocessed_chunk)
        online_sigs[engine] = np.concatenate(all_online_sigs)
        t2 = time.perf_counter()
        print(engine, 'process time', t2-t1)
    
    # remove border for comparison
    min_size = min([online_sigs[engine].shape[0] for engine in engines]) 
    offline_sig = offline_sig[chunksize:min_size]
    for engine in engines:
        online_sig = online_sigs[engine]
        online_sigs[engine] = online_sig[chunksize:min_size]

    # compare
    for engine in engines:
        online_sig = online_sigs[engine]
        residual = np.abs((online_sig.astype('float64')-offline_sig.astype('float64'))/np.mean(np.abs(offline_sig.astype('float64'))))

        print(np.max(residual))
        #~ print(np.mean(np.abs(offline_sig.astype('float64'))))
        assert np.max(residual)<5e-5, 'online differt from offline'
    
        # plot
        fig, axs = pyplot.subplots(nrows=nb_channel, sharex=True)
        for i in range(nb_channel):
            ax = axs[i]
            
            ax.plot(offline_sig[:, 2], color = 'g')
            ax.plot(online_sig[:, 2], color = 'r', ls='--')
            
            for i in range(nloop):
                ax.axvline(i*chunksize, color='k', alpha=0.4)
        
        pyplot.show()



    
if __name__ == '__main__':
    test_compare_offline_online_engines()
