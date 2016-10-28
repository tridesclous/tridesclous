import numpy as np


from collections import OrderedDict

try:
    from pyqtgraph.Qt import QtCore, QtGui
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


class SpikeSorter:
    """
    Main class for spike sorting that encaspulate all other class in the same place:
            * DataIO : acces unfiltered/unfiltred signals
            * SignalPreprocessing: preprocessing for signals (filter, bassline removal, normalise, whittening...)
            * CatalogueConstructor: construct the catalgue on a small chunk of dataset:
                * find peaks
                * extract waveforms
                * features (PCA, ...)
                * clustering (manual/auto)
            * TemplatePeeler: this is applied on the entire dataset:
                * Extract spike from signals 
            

        SpikeSorter handle the multi segment (several files).
        
        spikesorter = SpikeSorter()
    
    """
    def __init__(self, ):
        self.dataio = 
        self.signal_preprocessor = 
        self.catalogue_constructor = 
        self.template_peeler = 
        
    def summary(self, level=1):
        t = ''
        return t

    def __repr__(self):
        return self.summary(level=0)
    
    def run_preprocessing(self):
        pass
    
    def make_initial_catalogue(self):
        pass
        
    def run_template_peeler(self):
        pass
