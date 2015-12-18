from tridesclous import DataIO, SignalFilter

from matplotlib import pyplot


def test_filter():
    dataio = DataIO(dirname = 'datatest_neo')
    sigs = dataio.get_signals(seg_num=0, filtered = False)
    print(sigs)
    
    filter =  SignalFilter(sigs, highpass_freq = 300.)
    
    filterred_sigs = filter.get_filtered_data()
    
    sigs.plot()
    filterred_sigs.plot()
    
    


    

    
if __name__ == '__main__':
    test_filter()
    
    pyplot.show()