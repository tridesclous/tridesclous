import pytest
import os, tempfile, shutil
import numpy as np

from tridesclous import RawDataIO



def test_RawDataIO():
    
    
    # initialze dataio
    if os.path.exists('test_rawdataio'):
        shutil.rmtree('test_rawdataio')
        
    dataio = RawDataIO(dirname='test_rawdataio')
    filenames = ['Tem06c06.IOT', 'Tem06c07.IOT', 'Tem06c08.IOT']
    dataio.set_initial_signals(filenames=filenames, dtype='int16', total_channel=16, sample_rate=10000.)
    dataio.set_channel_group(range(14))
    
    for seg_num in range(dataio.nb_segment):
        #~ print('seg_num', seg_num)
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14
            #~ print(seg_num, i_stop, sigs_chunk.shape)
    
    #reopen existing
    dataio = RawDataIO(dirname='test_rawdataio')
    #~ print(dataio.info)
    for seg_num in range(dataio.nb_segment):
        #~ print('seg_num', seg_num)
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14
    
    
    
    
if __name__=='__main__':
    test_RawDataIO()
    
    