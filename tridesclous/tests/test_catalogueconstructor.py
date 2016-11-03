import numpy as np
import time

from tridesclous import get_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor



def test_catalogue_constructor():
    sigs, sample_rate = get_dataset()
    #~ sigs = sigs[:, [0]]
    nb_channel = sigs.shape[1]
    print('nb_channel', nb_channel)

    #params
    chunksize = 1024
    nloop = sigs.shape[0]//chunksize
    sigs = sigs[:chunksize*nloop]
    
    
    #TODO remove this just for testing
    dataio = DataIO()
    dataio.dtype = np.dtype('int16')
    dataio.sample_rate = sample_rate
    dataio.nb_channel = nb_channel
    
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    catalogueconstructor.initialize(chunksize=1024,
            memory_mode='ram',
            
            #signal preprocessor
            highpass_freq=300, backward_chunksize=1280,
            
            #peak detector
            peakdetector_engine='peakdetector_numpy',
            peak_sign='-', relative_threshold=5, peak_span=0.0005,
            
            #waveformextractor
            n_left=-20, n_right=30, 
            
            #features
            pca_batch_size=16384,
            )


    t1 = time.perf_counter()
    for i in range(nloop):
        #~ print()
        pos = (i+1)*chunksize
        #~ print('loop', i, 'pos', pos-chunksize, pos)
        
        chunk = sigs[pos-chunksize:pos,:]
        
        catalogueconstructor.process_one_chunk(pos, chunk)
        
    t2 = time.perf_counter()
    print('online process time', t2-t1)
    
    
    #~ process_one_chunk(self, pos, sigs_chunk)
    



    
if __name__ == '__main__':
    test_catalogue_constructor()
