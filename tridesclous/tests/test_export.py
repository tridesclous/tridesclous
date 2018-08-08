import os, shutil

import numpy as np

from tridesclous.export import export_csv, export_matlab, export_excel
from tridesclous.catalogueconstructor import _dtype_cluster
from tridesclous.peeler import _dtype_spike





def test_export():
    if os.path.exists('test_export'):
        shutil.rmtree('test_export')
    
    export_path = 'test_export'
    nspike = 50
    ncluster = 3
    
    seg_num = 0
    chan_grp = 0
    
    clusters = np.zeros(ncluster, dtype=_dtype_cluster)
    clusters['cluster_label'] = [0, 1, 4]
    clusters['cell_label'] = [0, 1, 1]
    
    catalogue = {}
    catalogue['clusters'] = clusters
    
    spikes = np.zeros(nspike, dtype=_dtype_spike)
    spikes['index'] = np.sort(np.random.randint(0, high=1000000, size=nspike))
    spikes['cluster_label'] = clusters['cluster_label'][np.random.randint(0, high=ncluster, size=nspike)]
    
    #~ print(clusters)
    #~ print(spikes)
    
    for split_by_cluster in (True, False):
        for use_cell_label in (True, False):
            
            args = (spikes, catalogue, seg_num, chan_grp, export_path+'/split{} celllabel{}/'.format(split_by_cluster, use_cell_label),)
            kargs = dict(split_by_cluster=split_by_cluster, use_cell_label=use_cell_label)
            
            export_csv(*args, **kargs)
            export_matlab(*args, **kargs)
            export_excel(*args, **kargs)
            
    
    
    
if __name__ == '__main__':
    test_export()