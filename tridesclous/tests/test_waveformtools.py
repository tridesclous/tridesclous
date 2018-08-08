import numpy as np
import time


from tridesclous.waveformtools import extract_chunks




def test_extract_chunks():
    size = 100000
    width = 50
    signals = np.random.randn(size, 5)
    indexes = np.random.randint(low=width, high=size-width, size=10)
    chunks = extract_chunks(signals, indexes, width, chunks=None)
    
    chunks[:] = 0
    chunks = extract_chunks(signals, indexes, width, chunks=chunks)
    

if __name__ == '__main__':
    test_extract_chunks()
    


