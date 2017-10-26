from tridesclous import *
import  pyqtgraph as pg
from matplotlib import pyplot

# run test_catalogueconstructor.py before this


def test_metrics():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    catalogueconstructor.compute_spike_waveforms_similarity()
    catalogueconstructor.compute_cluster_similarity()
    catalogueconstructor.compute_cluster_ratio_similarity()
    catalogueconstructor.compute_spike_silhouette()
    


if __name__ == '__main__':
    test_metrics()