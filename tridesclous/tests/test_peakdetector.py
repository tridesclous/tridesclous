from tridesclous import DataManager

from tridesclous import normalize_signals, derivative_signals, rectify_signals, detect_peak_method_span

from matplotlib import pyplot


def test_normalize_signals():
    data = DataManager(dirname = 'test')
    sigs = data.get_signals(seg_num=0)
    normed_sigs = normalize_signals(sigs)
    normed_sigs[3.14:3.22].plot()
    

def test_derivative_signals():
    data = DataManager(dirname = 'test')
    sigs = data.get_signals(seg_num=0)
    deriv_sigs = derivative_signals(sigs)
    deriv_sigs[3.14:3.22].plot()

def test_rectify_signals():
    data = DataManager(dirname = 'test')
    sigs = data.get_signals(seg_num=0)
    retified_sigs = rectify_signals(normalize_signals(sigs), threshold = -4)
    
    fig, ax = pyplot.subplots()
    retified_sigs[3.14:3.22].plot(ax = ax)
    ax.set_ylim(-20, 10)

def test_detect_peak_method_span():
    data = DataManager(dirname = 'test')
    sigs = data.get_signals(seg_num=0)
    normed_sigs = normalize_signals(sigs)
    retified_sigs = rectify_signals(normed_sigs, threshold = -4)    
    peaks_pos = detect_peak_method_span(retified_sigs,  peak_sign='-', n_span = 5)
    peaks_index = sigs.index[peaks_pos]
    
    fig, ax = pyplot.subplots()
    chunk = retified_sigs[3.14:3.22]
    chunk.plot(ax = ax)
    peaks_value = retified_sigs.loc[peaks_index]
    peaks_value[3.14:3.22].plot(marker = 'o', linestyle = 'None', ax = ax, color = 'k')
    ax.set_ylim(-20, 10)
    
    fig, ax = pyplot.subplots()
    chunk = normed_sigs[3.14:3.22]
    chunk.plot(ax = ax)
    peaks_value = normed_sigs.loc[peaks_index]
    peaks_value[3.14:3.22].plot(marker = 'o', linestyle = 'None', ax = ax, color = 'k')
    ax.set_ylim(-20, 10)
    

    
if __name__ == '__main__':
    test_normalize_signals()
    test_derivative_signals()
    test_rectify_signals()
    test_detect_peak_method_span()
    
    pyplot.show()
