import os
import shutil
from pprint import pprint

import numpy as np

from tridesclous.datasets import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.cataloguetools import apply_all_catalogue_steps
from tridesclous.autoparams import get_auto_params_for_catalogue

def is_running_on_ci_cloud():
    
    if os.environ.get('TRAVIS') in ('true', 'True'):
        return True
    
    if os.environ.get('APPVEYOR') in ('true', 'True'):
        return True

    if os.environ.get('CIRCLECI') in ('true', 'True'):
        return True
    
    return False

ON_CI_CLOUD = is_running_on_ci_cloud()
    
    

def setup_catalogue(dirname, dataset_name='olfactory_bulb', 
                duration=None, peak_sampler_mode=None):
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

        cluster_method = 'pruningshears'
        cluster_kargs = {'adjacency_radius_um' : 350 }
        
    else:
        channels = [0,1,2,3]
        mode = 'dense'
        adjacency_radius_um = None
        peak_method = 'global'
        peak_engine = 'numpy'
        
        feature_method = 'global_pca'
        feature_kargs = {'n_components': 6}

        cluster_method = 'pruningshears'
        cluster_kargs = {'adjacency_radius_um' : 150 }
        

    dataio.add_one_channel_group(channels=channels)
    
    
    cc = CatalogueConstructor(dataio=dataio)
    
    params = get_auto_params_for_catalogue(dataio)
    params['mode'] = mode
    params['n_jobs'] = 1
    params['peak_detector']['method'] = peak_method
    params['peak_detector']['engine'] = peak_engine
    params['peak_detector']['adjacency_radius_um'] = adjacency_radius_um
    params['feature_method'] = feature_method
    params['feature_kargs'] = feature_kargs
    params['cluster_method'] = cluster_method
    params['cluster_kargs'] = cluster_kargs
    
    if duration is not None:
        params['duration'] = duration
    
    if peak_sampler_mode is not None:
        params['peak_sampler']['mode'] = peak_sampler_mode
        
    
    #~ pprint(params)
    cc.apply_all_steps(params, verbose=True)
    
    # already done in apply_all_catalogue_steps:
    # cc.make_catalogue_for_peeler(inter_sample_oversampling=False, catalogue_name='initial') 
    
    # DEBUG
    #~ cc.make_catalogue_for_peeler(inter_sample_oversampling=True, catalogue_name='with_oversampling')
    
    return cc, params




if __name__ =='__main__':
    print('is_running_on_ci_cloud', is_running_on_ci_cloud())
    