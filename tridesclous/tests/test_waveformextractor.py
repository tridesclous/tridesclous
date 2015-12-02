from tridesclous import DataIO, PeakDetector

from tridesclous import extract_waveforms


from matplotlib import pyplot



def test_extract_waveform():
    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 2)
    
    
    #~ print(peakdetector.peak_pos)
    peak_pos = peak_pos[:200]
    waveforms = extract_waveforms(sigs, peak_pos, 15,20)
    #~ print(waveforms.columns)
    #~ print(waveforms.index)
    print(waveforms.shape)
    #~ fig, ax = pyplot.subplots()
    #~ waveforms.transpose().plot(ax =ax)
    
    fig, ax = pyplot.subplots()
    waveforms.median(axis=0).plot(ax =ax)



if __name__ == '__main__':
    test_extract_waveform()
    
    pyplot.show()
