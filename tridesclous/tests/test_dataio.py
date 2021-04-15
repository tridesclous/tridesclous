import pytest
import os, tempfile, shutil
import numpy as np

from tridesclous import download_dataset
from tridesclous import DataIO





    

def test_DataIO():
    
    
    # initialze dataio
    if os.path.exists('test_DataIO'):
        shutil.rmtree('test_DataIO')
        
    dataio = DataIO(dirname='test_DataIO')
    print(dataio)


    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    
    #with geometry
    channels = list(range(14))
    channel_groups = {0:{'channels':range(14), 'geometry' : { c: [0, i] for i, c in enumerate(channels) }}}
    dataio.set_channel_groups(channel_groups)
    
    #with no geometry
    channel_groups = {0:{'channels':range(4)}}
    dataio.set_channel_groups(channel_groups)
    
    # add one group
    dataio.add_one_channel_group(channels=range(4,8), chan_grp=5)
    
    
    channel_groups = {0:{'channels':range(14)}}
    dataio.set_channel_groups(channel_groups)
    
    for seg_num in range(dataio.nb_segment):
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14
            #~ print(seg_num, i_stop, sigs_chunk.shape)
    
    
    #reopen existing
    dataio = DataIO(dirname='test_DataIO')
    print(dataio)
    
    
    for seg_num in range(dataio.nb_segment):
        #~ print('seg_num', seg_num)
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14



def test_DataIO_probes():
    # initialze dataio
    if os.path.exists('test_DataIO'):
        shutil.rmtree('test_DataIO')
        
    dataio = DataIO(dirname='test_DataIO')
    print(dataio)


    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames,  **params)
    
    probe_filename = 'neuronexus/A4x8-5mm-100-400-413-A32.prb'
    dataio.download_probe(probe_filename)
    dataio.download_probe('neuronexus/A4x8-5mm-100-400-413-A32')
    
    #~ print(dataio.channel_groups)
    #~ print(dataio.channels)
    #~ print(dataio.info['probe_filename'])
    
    assert dataio.nb_channel(0) == 8
    assert probe_filename.split('/')[-1] == dataio.info['probe_filename']
    
    dataio = DataIO(dirname='test_DataIO')
    #~ print(dataio)
    


def test_dataio_catalogue():
    if os.path.exists('test_DataIO'):
        shutil.rmtree('test_DataIO')
    
    dataio = DataIO(dirname='test_DataIO')
    
    catalogue = {}
    catalogue['chan_grp'] = 0
    catalogue['centers0'] = np.ones((300, 12, 50))
    
    catalogue['n_left'] = -15
    catalogue['signal_preprocessor_params'] = {'highpass_freq' : 300.}
    
    dataio.save_catalogue(catalogue, name='test')
    
    c2 = dataio.load_catalogue(name='test', chan_grp=0)
    #~ print(c2)
    assert c2['n_left'] == -15
    assert np.all(c2['centers0']==1)
    assert catalogue['signal_preprocessor_params']['highpass_freq'] == 300.
    



if __name__=='__main__':
    
    test_DataIO()
    test_DataIO_probes()
    test_dataio_catalogue()
    
    