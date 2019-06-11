import os
import shutil

import numpy as np

from tridesclous.datasets import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.cataloguetools import apply_all_catalogue_steps


def is_running_on_ci_cloud():
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
        channels = [5, 6, 7, 8, 9]
    else:
        channels = [0,1,2,3]
    dataio.add_one_channel_group(channels=channels)
    
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    
    params = {
        'duration' : 60.,
        'preprocessor' : {
            'highpass_freq' : 300.,
            'chunksize' : 1024,
            'lostfront_chunksize' : 100,
        },
        'peak_detector' : {
            'peak_sign' : '-',
            'relative_threshold' : 7.,
            'peak_span_ms' : 0.5,
        },
        'extract_waveforms' : {
            'wf_left_ms' : -2.5,
            'wf_right_ms' : 4.0,
            'nb_max' : 10000,
        },
        'clean_waveforms' : {
            'alien_value_threshold' : 60.,
        },
        'noise_snippet' : {
            'nb_snippet' : 300,
        },
        'feature_method': 'global_pca', 
        'feature_kargs':{'n_components': 5},
        'cluster_method' : 'kmeans', 
        'cluster_kargs' : {'n_clusters': 12},
        'clean_cluster' : False,
        'clean_cluster_kargs' : {},
    }
    
    apply_all_catalogue_steps(catalogueconstructor, params, verbose=True)
        
    catalogueconstructor.trash_small_cluster()
    
    catalogueconstructor.order_clusters(by='waveforms_rms')
    
    
    catalogueconstructor.make_catalogue_for_peeler()




if __name__ =='__main__':
    print('is_running_on_ci_cloud', is_running_on_ci_cloud())
    