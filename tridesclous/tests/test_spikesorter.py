from tridesclous import SpikeSorter


def test_spikesorter():
    spikesorter = SpikeSorter(dirname = 'datatest')
    print(spikesorter)
    
    spikesorter.detect_peaks(seg_nums = 'all',  threshold=-4, peak_sign = '-', n_span = 2)
    
    print(spikesorter.summary(level=1))
    
    
    
if __name__ == '__main__':
    test_spikesorter()
