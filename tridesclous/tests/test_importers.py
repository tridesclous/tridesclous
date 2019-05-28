import os
import shutil
import pytest

from tridesclous import import_from_spykingcircus, import_from_spike_interface


@pytest.mark.skip()
def test_import_from_spykingcircus():
    p =  '/home/samuel/Documents/projet/DataSpikeSorting/Pierre/GT 252/'
    data_filename = p + '20160426/patch_2.raw'
    spykingcircus_dirname = p+'20160426/patch_2/'
    tdc_dirname = p+'spykingcircus_GT256_20160426'
    
    if os.path.exists(tdc_dirname):
        shutil.rmtree(tdc_dirname)
    
    cc = import_from_spykingcircus(data_filename, spykingcircus_dirname, tdc_dirname)
    
    print(cc.dataio)
    print(cc)

@pytest.mark.skip()
def test_import_from_spike_interface():
    import spikeextractors as se
    p = '/media/samuel/SamCNRS/DataSpikeSorting/mearec/'
    mearec_filename = p + 'recordings_50cells_SqMEA-10-15um_60.0_10.0uV_27-03-2019_13_31.h5'
    rec0  = se.MEArecRecordingExtractor(mearec_filename)
    
    for chan in rec0.get_channel_ids(): # remove3D
        loc = rec0.get_channel_property(chan, 'location')
        rec0.set_channel_property(chan, 'location', loc[1:])
    
    gt_sorting0 = se.MEArecSortingExtractor(mearec_filename)
    
    tdc_dirname = p + 'working_folder/output_folders/rec0/tridesclous/'

    if os.path.exists(tdc_dirname):
        shutil.rmtree(tdc_dirname)
    
    import_from_spike_interface(rec0, gt_sorting0, tdc_dirname)
    


if __name__ =='__main__':
    #~ test_import_from_spykingcircus()
    test_import_from_spike_interface()