from tridesclous import DataIO, SignalFilter

from matplotlib import pyplot


def test_filter():
    dataio = DataIO(dirname = 'datatest_neo')
    sigs = dataio.get_signals(seg_num=0, signal_type = 'unfiltered')
    
    filter =  SignalFilter(sigs, highpass_freq = 300.)
    filterred_sigs = filter.get_filtered_data()

    filter2 =  SignalFilter(sigs, highpass_freq = 300., box_smooth = 3)
    filterred_sigs2 = filter2.get_filtered_data()
    
    
    fig, axs = pyplot.subplots(nrows=3, sharex = True)
    sigs[0:.5].plot(ax=axs[0])
    filterred_sigs[0:.5].plot(ax=axs[1])
    filterred_sigs2[0:.5].plot(ax=axs[2])
    
if __name__ == '__main__':
    test_filter()
    
    pyplot.show()