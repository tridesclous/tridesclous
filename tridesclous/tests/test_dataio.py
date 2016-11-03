import pytest
import os, tempfile, shutil
import numpy as np

from tridesclous import RawDataIO



def test_RawDataIO():
    if os.path.exists('test_rawdataio'):
        shutil.rmtree('test_rawdataio')
    
    filenames = ['Tem06c06.IOT', 'Tem06c07.IOT', 'Tem06c08.IOT']
    
    dataio = RawDataIO(dirname='test_rawdataio', dtype='int16', nb_channel=16, sample_rate=10000.)
    
    
    
    
    
    
    
    
    
    
if __name__=='__main__':
    test_RawDataIO()
    
    