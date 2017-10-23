
from tridesclous import import_from_spykingcircus


def test_import_from_spykingcircus():
    p =  '/home/samuel/Documents/projet/DataSpikeSorting/Pierre/GT 252/'
    data_filename = p + '20160426/patch_2.raw'
    spykingcircus_dirname = p+'20160426/patch_2/'
    tdc_dirname = p+'spykingcircus_GT256_20160426'
    
    import_from_spykingcircus(data_filename, spykingcircus_dirname, tdc_dirname)




if __name__ =='__main__':
    test_import_from_spykingcircus()