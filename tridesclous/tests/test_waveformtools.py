import numpy as np
import time


from tridesclous.waveformtools import extract_chunks

#~ size = 1000000
size = 1000
nb_channel = 5
#~ nb_channel = 300
width = 150
nb_peak = 20003


def test_extract_chunks_memory():
    signals = np.random.randn(size, nb_channel).astype('float32')
    indexes = np.random.randint(low=width, high=size-width, size=nb_peak)

    t0 = time.perf_counter()
    chunks = extract_chunks(signals, indexes, width, chunks=None)
    t1 = time.perf_counter()
    print('extract_chunks no buffer', t1-t0)
    

    chunks[:] = 0
    
    t0 = time.perf_counter()
    chunks = extract_chunks(signals, indexes, width, chunks=chunks)
    t1 = time.perf_counter()
    print('extract_chunks with buffer', t1-t0)




def test_extract_chunks_memmap():
    signals = np.memmap('test_extract_wf_signal', dtype='float32', mode='w+', shape=(size, nb_channel))
    indexes = np.random.randint(low=width, high=size-width, size=nb_peak)
    chunks = np.memmap('test_extract_wf_chunks', dtype='float32', mode='w+', shape=(nb_peak, width, nb_channel))
    
    t0 = time.perf_counter()
    chunks = extract_chunks(signals, indexes, width, chunks=chunks)
    chunks.flush()
    t1 = time.perf_counter()
    print('extract_chunks memmap to memmap', t1-t0)


def test_extract_chunks_with_channel_indexes():
    signals = np.random.randn(size, nb_channel).astype('float32')
    indexes = np.random.randint(low=width, high=size-width, size=nb_peak)

    t0 = time.perf_counter()
    chunks = extract_chunks(signals, indexes, width, channel_indexes=[0,2,3], chunks=None)
    t1 = time.perf_counter()
    print('extract_chunks channel_indexes', t1-t0)



    

if __name__ == '__main__':
    test_extract_chunks_memory()
    test_extract_chunks_memmap()
    test_extract_chunks_with_channel_indexes()
    
    
    


