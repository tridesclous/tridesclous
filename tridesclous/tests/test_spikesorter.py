from tridesclous import SpikeSorter


def test_spikesorter():
    spikesorter = SpikeSorter(dirname = 'datatest')
    print(spikesorter)
    
    spikesorter.detect_peaks_extract_waveforms(seg_nums = 'all',  threshold=-4, peak_sign = '-', n_span = 2,  n_left=-30, n_right=50)
    print(spikesorter.summary(level=1))
    spikesorter.project(method = 'pca', n_components = 5)
    spikesorter.find_clusters(7)
    
    
    
    
    
if __name__ == '__main__':
    test_spikesorter()
