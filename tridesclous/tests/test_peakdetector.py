from tridesclous import DataManager

from tridesclous import normalize_signals, derivative_signals

from matplotlib import pyplot


def test_normalize_signals():
    data = DataManager(dirname = 'test')
    sigs = data.get_signals(seg_num=0)
    print(sigs.shape)
    normed_sigs = normalize_signals(sigs)
    
    fig, ax = pyplot.subplots()
    ax.plot(normed_sigs.index, normed_sigs)
    pyplot.show()

def test_derivative_signals():
    data = DataManager(dirname = 'test')
    sigs = data.get_signals(seg_num=0)
    print(sigs.shape)
    normed_sigs = derivative_signals(sigs)
    print(normed_sigs.shape)
    fig, ax = pyplot.subplots()
    ax.plot(normed_sigs.index, normed_sigs)
    pyplot.show()

    
    
if __name__ == '__main__':
    #~ test_normalize_signals()
    test_derivative_signals()