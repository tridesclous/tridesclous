import shutil

from tridesclous.dataio import DataIO

from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.tests.testingtools import setup_catalogue
from tridesclous.tests.testingtools import ON_CI_CLOUD

from tridesclous.report import summary_catalogue_clusters, summary_noise, summary_after_peeler_clusters


import matplotlib.pyplot as plt


def setup_module():
    setup_catalogue('test_report', dataset_name='striatum_rat')

def teardown_module():
    shutil.rmtree('test_report')

def test_summary_catalogue_clusters():
    dataio = DataIO(dirname='test_report')

    #~ summary_catalogue_clusters(dataio, chan_grp=0)
    summary_catalogue_clusters(dataio, chan_grp=0, labels= [0])
    
def test_summary_noise():
    dataio = DataIO(dirname='test_report')
 
    
    summary_noise(dataio, chan_grp=0)
    
def test_summary_after_peeler_clusters():
    #~ dataio = DataIO(dirname='test_report')

    workdir = '/home/samuel/Documents/projet/DataSpikeSorting/kampff/polytrode Impedance/'
    dirname = workdir + 'tdc_amplifier2017-02-02T17_18_46'
    dataio = DataIO(dirname)
    
    summary_after_peeler_clusters(dataio, chan_grp=0,  labels= [0])

if __name__ == '__main__':
    
    #~ setup_module()
    
    #~ test_summary_catalogue_clusters()
    
    #~ test_summary_noise()
    
    test_summary_after_peeler_clusters()
    
    
    plt.show()
    