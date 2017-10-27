from tridesclous import *
import  pyqtgraph as pg
import matplotlib.pyplot as plt

# run test_catalogueconstructor.py before this


def test_all_metrics():
    dataio = DataIO(dirname='test_catalogueconstructor')
    cc = CatalogueConstructor(dataio=dataio)
    
    cc.compute_spike_waveforms_similarity()
    cc.compute_cluster_similarity()
    cc.compute_cluster_ratio_similarity()
    cc.compute_spike_silhouette()


def test_cluster_ratio():
    dataio = DataIO(dirname='test_catalogueconstructor')
    cc = CatalogueConstructor(dataio=dataio)
    
    cc.compute_cluster_similarity()
    cc.compute_cluster_ratio_similarity()

    for name in('cluster_similarity', 'cluster_ratio_similarity'):
        d = getattr(cc, name)
        fig, ax = plt.subplots()
        im  = ax.matshow(d, cmap='viridis')
        fig.colorbar(im)
        ax.set_title(name)

    
    plt.show()


if __name__ == '__main__':
    #~ test_all_metrics()
    test_cluster_ratio()