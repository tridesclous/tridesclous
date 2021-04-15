import os
from collections import OrderedDict

import numpy as np
import scipy.io

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class GenericSpikeExporter:
    def __call__(self,spikes, catalogue, seg_num, chan_grp, export_path,
                        split_by_cluster=False,
                        use_cell_label=True,
                        #~ use_index=True,
                        ):
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        #~ print('export', spikes.size, seg_num, export_path)
        #~ print('split_by_cluster', split_by_cluster, 'use_cell_label', use_cell_label)
        clusters = catalogue['clusters']

        spike_labels = spikes['cluster_label']
        if use_cell_label:
            spike_labels = spikes['cluster_label'].copy()
            for l in clusters:
                mask = spike_labels==l['cluster_label']
                spike_labels[mask] = l['cell_label']

        spike_indexes = spikes['index']


        out_data = OrderedDict()
        if split_by_cluster:
            if use_cell_label:
                possible_labels = np.unique(clusters['cell_label'])
                label_name = 'cell'
            else:
                possible_labels = clusters['cluster_label']
                label_name = 'cluster'
            for k in possible_labels:
                keep = k == spike_labels
                out_data[label_name + '#'+ str(k)] = (spike_indexes[keep], spike_labels[keep])
        else:
            out_data['cell#all'] = (spike_indexes, spike_labels)

        name = 'spikes - segNum {} - chanGrp {}'.format(seg_num, chan_grp)
        filename = os.path.join(export_path, name)
        self.write_out_data(out_data, filename)

class CsvSpikeExporter(GenericSpikeExporter):
    ext = 'csv'
    def write_out_data(self, out_data, filename):
        for key, (spike_indexes, spike_labels) in out_data.items():
            filename2 = filename +' - '+key+'.csv'
            self._write_one_file(filename2, spike_indexes, spike_labels)

    def _write_one_file(self, filename, labels, indexes):
        rows = [''] * len(labels)
        for i in range(len(labels)):
            rows[i]='{},{}\n'.format(labels[i], indexes[i])
        with open(filename, 'w') as out:
            out.writelines(rows)

export_csv = CsvSpikeExporter()


class MatlabSpikeExporter(GenericSpikeExporter):
    ext = 'mat'

    def write_out_data(self, out_data, filename):
        mdict = {}

        for key, (spike_indexes, spike_labels) in out_data.items():
            mdict['index_'+key] = spike_indexes
            mdict['label_'+key] =spike_labels
        scipy.io.savemat(filename+'.mat', mdict)

export_matlab = MatlabSpikeExporter()


class ExcelSpikeExporter(GenericSpikeExporter):
    ext = 'xslx'
    def write_out_data(self, out_data, filename):
        assert HAS_PANDAS
        writer = pd.ExcelWriter(filename+'.xlsx')
        for key, (spike_indexes, spike_labels) in out_data.items():
            df = pd.DataFrame()
            df['index'] = spike_indexes
            df['label'] = spike_labels
            df.to_excel(writer, sheet_name=key, index=False)
        writer.save()

export_excel = ExcelSpikeExporter()


# list
export_list = [export_csv, export_matlab, ]
if HAS_PANDAS:
    export_list.append(export_excel)

export_dict = {e.ext:e for e in export_list}


def export_catalogue_spikes(cc, export_path=None, formats=None):
    """
    This export spikes from catalogue.
    Usefull when when catalogue peak sampler mode is all.
    This avoid the peeler.
    """
    
    dataio = cc.dataio
    chan_grp = cc.chan_grp
    
    sampler_mode = cc.info['peak_sampler']['mode']
    if sampler_mode != 'all':
        print('You are trying to export peak from catalogue but peak_sampler mode is not "all"')

    if export_path is None:
        export_path = os.path.join(dataio.dirname, 'export_catalogue_chan_grp_{}'.format(chan_grp))
    
    catalogue = {}
    catalogue['clusters'] = cc.clusters.copy()
    
    if formats is None:
        exporters = export_list
    elif isinstance(formats, str):
        assert formats in export_dict
        exporters = [ export_dict[formats] ]
    elif isinstance(format, list):
        exporters = [ export_dict[format] for format in formats]
    else:
        raise ValueError()
        
    for seg_num in range(dataio.nb_segment):
        in_segment = (cc.all_peaks['segment'] == seg_num)
        pos_label = (cc.all_peaks['cluster_label'] >= 0)
        spikes = cc.all_peaks[in_segment & pos_label]
        
        if spikes is None: continue
        
        args = (spikes, catalogue, seg_num, chan_grp, export_path,)
        kargs = dict(split_by_cluster=False, use_cell_label=False)
        for exporter in exporters:
            exporter(*args, **kargs)
