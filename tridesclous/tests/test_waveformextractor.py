import pandas as pd
import numpy as np
from tridesclous import DataIO, PeakDetector, normalize_signals

from tridesclous import extract_peak_waveforms, extract_noise_waveforms,good_events,find_good_limits


from matplotlib import pyplot



def test_extract_peak_waveforms():
    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 2)
    
    waveforms = extract_peak_waveforms(sigs, peak_pos, 15,50)
    print(waveforms.shape)
    fig, ax = pyplot.subplots()
    waveforms.median(axis=0).plot(ax =ax)
    
    normed_waveforms = extract_peak_waveforms(peakdetector.normed_sigs, peak_pos, 15,50)
    fig, ax = pyplot.subplots()
    normed_waveforms.median(axis=0).plot(ax =ax)
    


def test_extract_noise_waveforms():

    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 2)
    
    
    waveforms = extract_noise_waveforms(sigs, peak_pos, 15,20, size = 500)
    print(waveforms.shape)
    
    fig, ax = pyplot.subplots()
    waveforms.median(axis=0).plot(ax =ax)


def test_good_events():
    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 2)
    
    #~ peak_pos = peak_pos[:100]
    #~ print(peak_pos)
    
    waveforms = extract_peak_waveforms(sigs, peak_pos, 15,20)
    keep = good_events(waveforms,upper_thr=5.,lower_thr=-5.)
    #~ print(keep)
    goods_wf = waveforms[keep]
    bads_wf = waveforms[~keep]


    fig, ax = pyplot.subplots()
    #~ goods_wf.transpose().plot(ax =ax, color = 'g', lw = .3)
    bads_wf.transpose().plot(ax =ax,color = 'r', lw = .3)
    
    
    med = waveforms.median(axis=0)
    mad = np.median(np.abs(waveforms-med),axis=0)*1.4826
    limit1 = med+5*mad
    limit2 = med-5*mad
    
    med.plot(ax = ax, color = 'm', lw = 2)
    limit1.plot(ax = ax, color = 'm')
    limit2.plot(ax = ax, color = 'm')
    

def test_find_good_limits():
    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    
    
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 5)
    
    
    normed_waveforms = extract_peak_waveforms(peakdetector.normed_sigs, peak_pos, 25,50)
    
    normed_med = normed_waveforms.median(axis=0)
    normed_mad = np.median(np.abs(normed_waveforms-normed_med),axis=0)*1.4826
    normed_mad = normed_mad.reshape(4,-1)
    
    fig, ax = pyplot.subplots()
    ax.plot(normed_mad.transpose())
    ax.axhline(1.1)
    
    l1, l2 = find_good_limits(normed_mad)
    ax.axvline(l1)
    ax.axvline(l2)
    
    
    
    
    


if __name__ == '__main__':
    #~ test_extract_peak_waveforms()
    #~ test_extract_noise_waveforms()
    #~ test_good_events()
    test_find_good_limits()
    
    
    
    pyplot.show()
