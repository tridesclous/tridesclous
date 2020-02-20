import os
import shutil
from pprint import pprint

import numpy as np

from tridesclous.datasets import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.cataloguetools import apply_all_catalogue_steps


def is_running_on_ci_cloud():
    return True
    if os.environ.get('TRAVIS') in ('true', 'True'):
        return True
    
    if os.environ.get('APPVEYOR') in ('true', 'True'):
        return True

    if os.environ.get('CIRCLECI') in ('true', 'True'):
        return True
    
    return False

ON_CI_CLOUD = is_running_on_ci_cloud()
    
    

def setup_catalogue(dirname, dataset_name='olfactory_bulb'):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
        
    dataio = DataIO(dirname=dirname)
    localdir, filenames, params = download_dataset(name=dataset_name)
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    
    if dataset_name=='olfactory_bulb':
        channels = [4, 5, 6, 7, 8, 9]
        mode = 'sparse'
        adjacency_radius_um = 350
        peak_method = 'geometrical'
        peak_engine = 'numpy'
        
        feature_method = 'pca_by_channel'
        feature_kargs = {'n_components_by_channel': 3}
    else:
        channels = [0,1,2,3]
        mode = 'dense'
        adjacency_radius_um = None
        peak_method = 'global'
        peak_engine = 'numpy'
        
        feature_method = 'global_pca'
        feature_kargs = {'n_components': 5}

    dataio.add_one_channel_group(channels=channels)
    
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    
    params = {
        'duration' : 60.,
        'chunksize': 1024,
        'mode': mode,
        'memory_mode': 'memmap',
        
        'preprocessor' : {
            'highpass_freq' : 300.,
            'lostfront_chunksize' : 100,
            'engine' : 'numpy',
        },
        'peak_detector' : {
            'peak_sign' : '-',
            'relative_threshold' : 7.,
            'peak_span_ms' : 0.5,
            'method' : peak_method,
            'engine' : peak_engine,
            'adjacency_radius_um':adjacency_radius_um,
        },
        'extract_waveforms' : {
            'wf_left_ms' : -2.5,
            'wf_right_ms' : 4.0,
            #~ 'nb_max' : 10000,
        },
        'clean_peaks' : {
            'alien_value_threshold' : 60.,
            'mode': 'full_waveform',
        },
        'peak_sampler':{
            'mode': 'rand',
            'nb_max' : 10000,
        },
        'noise_snippet' : {
            'nb_snippet' : 300,
        },
        'feature_method': feature_method,
        'feature_kargs':feature_kargs,
        #~ 'cluster_method' : 'kmeans', 
        #~ 'cluster_kargs' : {'n_clusters': 12},
        'cluster_method' : 'pruningshears', 
        'cluster_kargs' : {},
        'clean_cluster' : False,
        'clean_cluster_kargs' : {},
    }
    
    #~ pprint(params)
    apply_all_catalogue_steps(catalogueconstructor, params, verbose=True)
    
    catalogueconstructor.make_catalogue_for_peeler()




if __name__ =='__main__':
    print('is_running_on_ci_cloud', is_running_on_ci_cloud())
    