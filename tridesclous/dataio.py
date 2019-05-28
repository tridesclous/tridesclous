"""

.. autoclass:: DataIO
   :members:

"""

import os, shutil
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
import pickle

from .datasource import data_source_classes
from .iotools import ArrayCollection
from .tools import download_probe, create_prb_file_from_dict, fix_prb_file_py2

from .export import export_list, export_dict

_signal_types = ['initial', 'processed']





class DataIO:
    """
    Class to acces the dataset (raw data, processed, catalogue, 
    spikes) in read/write mode.
    
    All operations on the dataset are done througth that class.
    
    The dataio :
      * work in a path. Almost everything is persistent.
      * needed by CatalogueConstructor and Peeler
      * have a datasource member that access raw data
      * store/load processed signals
      * store/load spikes
      * store/load the catalogue
      * deal with sevral channel groups given a PRB file
      * deal with several segment of recording (aka several files for raw data)
      * export the results (labeled spikes) to differents format
      
    
    The underlying data storage is a simple tree on the file system.
    Everything is organised as simple as possible in sub folder 
    (by channel group then segment index).
    In each folder:
      * arrays.json describe the list of numpy array (name, dtype, shape)
      * XXX.raw are the raw numpy arrays and load with a simple memmap.
      * some array are struct arrays (aka array of struct)
      
    The datasource system is based on neo.rawio so all format in neo.rawio are
    available in tridesclous. neo.rawio is able to read chunk of signals indexed 
    on time axis and channel axis.
    
    The raw dataset do not need to be inside the working directory but can be somewhere outside.
    The info.json describe the link to the *datasource* (raw data)
    
    Many raw dataset are saved by the device with an underlying int16.
    DataIO save the processed signals as float32 by default. So if
    you have a 10Go raw dataset tridesclous will need at least 20 Go more for storage
    of the processed signals.
    
    
    **Usage**::
    
        # initialize a directory
        dataio = DataIO(dirname='/path/to/a/working/dir')
        
        # set a data source
        filenames = ['file1.raw', 'file2.raw']
        dataio.dataio.set_data_source(type='RawData', filenames=filenames, 
                                    sample_rate=10000, total_channel=16, dtype='int16')
        
        # set a PRB file
        dataio.set_probe_file('/path/to/a/file.prb')
        # or dowload it
        dataio.download_probe('kampff_128', origin='spyking-circus')
        
        # check lenght and channel groups
        print(dataio)

    
    """
    
    @staticmethod
    def check_initialized(dirname):
        if not os.path.exists(dirname):
            return False

        info_filename = os.path.join(dirname, 'info.json')
        if not os.path.exists(info_filename):
            return False
        
        return True
    
    def __init__(self, dirname='test'):
        self.dirname = dirname
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        self.info_filename = os.path.join(self.dirname, 'info.json')
        if not os.path.exists(self.info_filename):
            #first init
            self.info = {}
            self.flush_info()
            self.datasource = None
        else:
            with open(self.info_filename, 'r', encoding='utf8') as f:
                self.info = json.load(f)
            #~ print('*'*50)
            #~ print(self.info_filename)
            #~ print(self.info)
            #~ print('*'*50)
            #~ try:
            #~ if 1:
            if len(self.info)>0:
                #~ self._reload_info()
                self._reload_channel_group()
                self._reload_data_source()
                self._open_processed_data()
            else:
                self.datasource = None
                
            #~ except:
                #~ self.info = {}
                #~ self.flush_info()
                #~ self.datasource = None
 
    def __repr__(self):
        t = "DataIO <id: {}> \n  workdir: {}\n".format(id(self), self.dirname)
        if len(self.info) ==0 or self.datasource is None:
            t  += "\n  Not datasource is set yet"
            return t
        t += "  sample_rate: {}\n".format(self.sample_rate)
        t += "  total_channel: {}\n".format(self.total_channel)
        if len(self.channel_groups)==1:
            k0, cg0 = next(iter(self.channel_groups.items()))
            ch_names = np.array(self.datasource.get_channel_names())[cg0['channels']]
            if len(ch_names)>8:
                chantxt = "[{} ... {}]".format(' '.join(ch_names[:4]),' '.join(ch_names[-4:]))
            else:
                chantxt = "[{}]".format(' '.join(ch_names))
            t += "  channel_groups: {} {}\n".format(k0, chantxt)
        else:
            t += "  channel_groups: {}\n".format(', '.join(['{} ({}ch)'.format(cg, self.nb_channel(cg))
                                                                for cg in self.channel_groups.keys() ]))
        t += "  nb_segment: {}\n".format(self.nb_segment)
        if self.nb_segment<5:
            lengths = [ self.datasource.get_segment_shape(i)[0] for i in range(self.nb_segment)]
            t += '  length: '+' '.join('{}'.format(l) for l in lengths)+'\n'
            t += '  durations: '+' '.join('{:0.1f}'.format(l/self.sample_rate) for l in lengths)+' s.\n'
        if t.endswith('\n'):
            t = t[:-1]
        
        return t

 
    def flush_info(self):
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f, indent=4)
    
    def set_data_source(self, type='RawData', **kargs):
        """
        Set the datasource. Must be done only once otherwise raise error.
        
        
        Parameters
        ------------------
        
        type: str ('RawData', 'Blackrock', 'Neuralynx', ...)
            The name of the neo.rawio class used to open the dataset.
        kargs: 
            depends on the class used. They are transimted to neo class.
            So see neo doc for kargs
        
        """
        assert type in data_source_classes, 'this source type do not exists yet!!'
        assert 'datasource_type' not in self.info, 'datasource is already set'
        # force abs path name 
        if 'filenames' in kargs:
            kargs['filenames'] = [ os.path.abspath(f) for f in kargs['filenames']]
        if 'dirnames' in kargs:
            kargs['dirnames'] = [ os.path.abspath(f) for f in kargs['dirnames']]

        self.info['datasource_type'] = type
        self.info['datasource_kargs'] = kargs
        self._reload_data_source()
        # be default chennel group all channels
        channel_groups = {0:{'channels':list(range(self.total_channel))}}
        self.set_channel_groups( channel_groups, probe_filename='default.prb')
        
        self.flush_info()
        # this create segment path
        self._open_processed_data()
    
    def _reload_data_source(self):
        assert 'datasource_type' in self.info
        kargs = self.info['datasource_kargs']
        
        self.datasource = data_source_classes[self.info['datasource_type']](**kargs)
        
        self.total_channel = self.datasource.total_channel
        self.nb_segment = self.datasource.nb_segment
        self.sample_rate = self.datasource.sample_rate
        self.source_dtype = self.datasource.dtype
    
    def _reload_channel_group(self):
        #TODO test in prb is compatible with py3
        d = {}
        probe_filename = os.path.join(self.dirname, self.info['probe_filename'])
        with open(probe_filename) as f:
            exec(f.read(), None, d)
        channel_groups = d['channel_groups']

        for chan_grp, channel_group in channel_groups.items():
            assert 'channels' in channel_group
            channel_group['channels'] = list(channel_group['channels'])
            if 'geometry' not in channel_group or channel_group['geometry'] is None:
                channels = channel_group['channels']
                geometry = self._make_fake_geometry(channels)
                channel_group['geometry'] = geometry
        self.channel_groups = channel_groups
    
    def _rm_old_probe_file(self):
        old_filename = self.info.get('probe_filename', None)
        if old_filename is not None:
            os.remove(os.path.join(self.dirname, old_filename))
        
    def set_probe_file(self, src_probe_filename):
        """
        Set the probe file.
        The probe file is copied inside the working dir.
        
        """
        self._rm_old_probe_file()
        probe_filename = os.path.join(self.dirname, os.path.basename(src_probe_filename))
        try:
            shutil.copyfile(src_probe_filename, probe_filename)
        except shutil.SameFileError:
            # print('probe allready in dir')
            pass
        fix_prb_file_py2(probe_filename)
        # check that the geometry is 2D
        with open(probe_filename) as f:
            d = {}
            exec(f.read(), None, d)
            channel_groups = d['channel_groups']
            for chan_grp, channel_group in channel_groups.items():
                geometry = channel_group.get('geometry', None)
                if geometry is not None:
                    for c, v in geometry.items():
                        assert len(v) == 2, 'Tridesclous need 2D geometry'
                    
                
        
        self.info['probe_filename'] = os.path.basename(probe_filename)
        self.flush_info()
        self._reload_channel_group()
        self._open_processed_data()
        
    
    def download_probe(self, probe_name, origin='kwikteam'):
        """
        Download a prb file from github  into the working dir.
        
        The spiking-circus and kwikteam propose a list prb file.
        See:
           * https://github.com/kwikteam/probes
           * https://github.com/spyking-circus/spyking-circus/tree/master/probes
        
        Parameters
        ------------------
        probe_name: str
            the name of file in github
        origin: 'kwikteam' or 'spyking-circus'
            github project
        
        """
        self._rm_old_probe_file()
        
        
        probe_filename = download_probe(self.dirname, probe_name, origin=origin)
        
        #~ if origin == 'kwikteam':
            #~ #Max Hunter made a list of neuronexus probes, many thanks
            #~ baseurl = 'https://raw.githubusercontent.com/kwikteam/probes/master/'
        #~ elif origin == 'spyking-circus':
            #~ # Pierre Yger made a list of various probe file, many thanks
            #~ baseurl = 'https://raw.githubusercontent.com/spyking-circus/spyking-circus/master/probes/'
        #~ else:
            #~ raise(NotImplementedError)
        
        
        #~ if not probe_name.endswith('.prb'):
            #~ probe_name += '.prb'
        #~ probe_filename = os.path.join(self.dirname,probe_name)
        #~ urlretrieve(baseurl+probe_name, probe_filename)
        #~ fix_prb_file_py2(probe_filename)#fix range to list(range
        fix_prb_file_py2(probe_filename)
        self.info['probe_filename'] = os.path.basename(probe_filename)
        self.flush_info()
        self._reload_channel_group()
        self._open_processed_data()
    
    def _make_fake_geometry(self, channels):
        if len(channels)!=4:
            # assume that it is a linear probes with 200 um
            geometry = { c: [0, i*200] for i, c in enumerate(channels) }
        else:
            # except for tetrode
            geometry = dict(zip(channels, [(0., 50.), (50., 0.), (0., -50.), (-50, 0.)]))
        return geometry
        
    def set_channel_groups(self, channel_groups, probe_filename='channels.prb'):
        """
        Set manually the channel groups dictionary.
        """
        self._rm_old_probe_file()
        
        # checks
        for chan_grp, channel_group in channel_groups.items():
            assert 'channels' in channel_group
            channel_group['channels'] = list(channel_group['channels'])
            if 'geometry' not in channel_group or channel_group['geometry'] is None:
                channels = channel_group['channels']
                #~ geometry = { c: [0, i] for i, c in enumerate(channels) }
                geometry = self._make_fake_geometry(channels)
                channel_group['geometry'] = geometry
        
        # write with hack on json to put key as inteteger (normally not possible in json)
        #~ with open(os.path.join(self.dirname,probe_filename) , 'w', encoding='utf8') as f:
            #~ txt = json.dumps(channel_groups,indent=4)
            #~ for chan_grp in channel_groups.keys():
                #~ txt = txt.replace('"{}":'.format(chan_grp), '{}:'.format(chan_grp))
                #~ for chan in channel_groups[chan_grp]['channels']:
                    #~ txt = txt.replace('"{}":'.format(chan), '{}:'.format(chan))
            #~ txt = 'channel_groups = ' +txt
            #~ f.write(txt)
        
        create_prb_file_from_dict(channel_groups, os.path.join(self.dirname,probe_filename))
        
        
        
        self.info['probe_filename'] = probe_filename
        self.flush_info()
        self._reload_channel_group()
        self._open_processed_data()
    
    def add_one_channel_group(self, channels=[], chan_grp=0, geometry=None):
        channels = list(channels)
        if geometry is None:
            geometry = self._make_fake_geometry(channels)
        
        self.channel_groups[chan_grp] = {'channels': channels, 'geometry':geometry}
        #rewrite with same name
        self.set_channel_groups(self.channel_groups, probe_filename=self.info['probe_filename'])
        
    def get_geometry(self, chan_grp=0):
        """
        Get the geometry for a given channel group in a numpy array way.
        """
        channel_group = self.channel_groups[chan_grp]
        geometry = [ channel_group['geometry'][chan] for chan in channel_group['channels'] ]
        geometry = np.array(geometry, dtype='float64')
        return geometry

    def nb_channel(self, chan_grp=0):
        """
        Number of channel for a channel group.
        """
        #~ print('DataIO.nb_channel', self.channel_groups)
        return len(self.channel_groups[chan_grp]['channels'])


    def channel_group_label(self, chan_grp=0):
        """
        Label of channel for a group.
        """
        label = 'chan_grp {} - '.format(chan_grp)
        
        channels = self.channel_groups[chan_grp]['channels']
        ch_names = np.array(self.datasource.get_channel_names())[channels]
        
        if len(ch_names)<8:
            label += ' '.join(ch_names)
        else:
            label += ' '.join(ch_names[:3]) + ' ... ' + ' '.join(ch_names[-2:])
        
        return label


    
    def _open_processed_data(self):
        self.channel_group_path = {}
        self.segments_path = {}
        for chan_grp in self.channel_groups.keys():
            self.segments_path[chan_grp] = []
            cg_path = os.path.join(self.dirname, 'channel_group_{}'.format(chan_grp))
            self.channel_group_path[chan_grp] = cg_path
            if not os.path.exists(cg_path):
                os.mkdir(cg_path)
            for i in range(self.nb_segment):
                segment_path = os.path.join(cg_path, 'segment_{}'.format(i))
                if not os.path.exists(segment_path):
                    os.mkdir(segment_path)
                self.segments_path[chan_grp].append(segment_path)
        
        self.arrays = {}
        for chan_grp in self.channel_groups.keys():
            self.arrays[chan_grp] = []
            
            for i in range(self.nb_segment):
                arrays = ArrayCollection(parent=None, dirname=self.segments_path[chan_grp][i])
                self.arrays[chan_grp].append(arrays)
            
                for name in ['processed_signals', 'spikes']:
                    self.arrays[chan_grp][i].load_if_exists(name)
    
    def get_segment_length(self, seg_num):
        """
        Segment length (in sample) for a given segment index
        """
        full_shape =  self.datasource.get_segment_shape(seg_num)
        return full_shape[0]
    
    def get_segment_shape(self, seg_num, chan_grp=0):
        """
        Segment shape for a given segment index and channel group.
        """
        full_shape =  self.datasource.get_segment_shape(seg_num)
        shape = (full_shape[0], self.nb_channel(chan_grp))
        return shape
    
    def get_signals_chunk(self, seg_num=0, chan_grp=0,
                i_start=None, i_stop=None,
                signal_type='initial'): #return_type='raw_numpy'
        """
        Get a chunk of signal for for a given segment index and channel group.
        
        The signal can be the 'initial' (aka raw signal), the none filetered signals or
        the 'processed' signal.
        
        Parameters
        ------------------
        seg_num: int
            segment index
        chan_grp: int
            channel group key
        i_start: int or None
           start index (included)
        i_stop: int or None
            stop index (not included)
        signal_type: str
            'initial' or 'processed'
        
        
        
        """
        channels = self.channel_groups[chan_grp]['channels']
        
        if signal_type=='initial':
            data = self.datasource.get_signals_chunk(seg_num=seg_num, i_start=i_start, i_stop=i_stop)
            data = data[:, channels]
        elif signal_type=='processed':
            data = self.arrays[chan_grp][seg_num].get('processed_signals')[i_start:i_stop, :]
        else:
            raise(ValueError, 'signal_type is not valide')
        
        return data
        #~ if return_type=='raw_numpy':
            #~ return data
        #~ elif return_type=='on_scale_numpy':
            #~ raise(NotImplementedError)
        #~ elif return_type=='pandas':
            #~ raise(NotImplementedError)

    def iter_over_chunk(self, seg_num=0, chan_grp=0,  i_stop=None, chunksize=1024, **kargs):
        """
        Create an iterable on signals. ('initial' or 'processed')
        
        Usage
        ----------
        
            for ind, sig_chunk in data.iter_over_chunk(seg_num=0, chan_grp=0, chunksize=1024, signal_type='processed'):
                do_something_on_chunk(sig_chunk)
        
        """
        if i_stop is not None:
            length = min(self.get_segment_shape(seg_num, chan_grp=chan_grp)[0], i_stop)
        else:
            length = self.get_segment_shape(seg_num, chan_grp=chan_grp)[0]
        
        
        nloop = length//chunksize
        for i in range(nloop):
            i_stop = (i+1)*chunksize
            i_start = i_stop - chunksize
            sigs_chunk = self.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp, i_start=i_start, i_stop=i_stop, **kargs)
            yield  i_stop, sigs_chunk
        
        # lat chunk = very bad idea because it break chunksize in many place (OpenCL signal processor, Peeler, ...)
        #~ if i_stop<length:
            #~ i_start = i_stop
            #~ i_stop = length
            #~ sigs_chunk = self.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp, i_start=i_start, i_stop=i_stop, **kargs)
            #~ yield  i_stop, sigs_chunk
    
    def reset_processed_signals(self, seg_num=0, chan_grp=0, dtype='float32'):
        """
        Reset processed signals.
        """
        self.arrays[chan_grp][seg_num].create_array('processed_signals', dtype, 
                            self.get_segment_shape(seg_num, chan_grp=chan_grp), 'memmap')
    
    def set_signals_chunk(self,sigs_chunk, seg_num=0, chan_grp=0, i_start=None, i_stop=None, signal_type='processed'):
        """
        Set a signal chunk (only for 'processed')
        """
        assert signal_type != 'initial'

        if signal_type=='processed':
            data = self.arrays[chan_grp][seg_num].get('processed_signals')
            data[i_start:i_stop, :] = sigs_chunk
        
    def flush_processed_signals(self, seg_num=0, chan_grp=0):
        """
        Flush the underlying memmap for processed signals.
        """
        self.arrays[chan_grp][seg_num].flush_array('processed_signals')
    
    def reset_spikes(self, seg_num=0,  chan_grp=0, dtype=None):
        """
        Reset spikes.
        """
        assert dtype is not None
        self.arrays[chan_grp][seg_num].initialize_array('spikes', 'memmap', dtype, (-1,))
        
    def append_spikes(self, seg_num=0, chan_grp=0, spikes=None):
        """
        Append spikes.
        """
        if spikes is None: return
        self.arrays[chan_grp][seg_num].append_chunk('spikes', spikes)
        
    def flush_spikes(self, seg_num=0, chan_grp=0):
        """
        Flush underlying memmap for spikes.
        """
        self.arrays[chan_grp][seg_num].finalize_array('spikes')
    
    def is_spike_computed(self, chan_grp=0):
        done = all(self.arrays[chan_grp][seg_num].has_key('spikes') for seg_num in range(self.nb_segment))
        return done
    
    def get_spikes(self, seg_num=0, chan_grp=0, i_start=None, i_stop=None):
        """
        Read spikes
        """
        if not self.arrays[chan_grp][seg_num].has_key('spikes'):
            return None
        spikes = self.arrays[chan_grp][seg_num].get('spikes')
        if spikes is None:
            return
        return spikes[i_start:i_stop]
    
    def get_some_waveforms(self, seg_num=0, chan_grp=0, spike_indexes=None, n_left=None, n_right=None):
        """
        Exctract some waveforms given spike_indexes
        """
        assert spike_indexes is not None, 'Provide spike_indexes'
        peak_width = n_right - n_left
        wf = np.zeros((spike_indexes.size, peak_width, self.nb_channel(chan_grp)), dtype='float32')
        for i, spike_index in enumerate(spike_indexes):
            i_start = spike_index + n_left
            i_stop = spike_index + n_right
            wf[i, :, :] = self.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp, 
                        i_start=i_start, i_stop=i_stop, signal_type='processed')
        return wf
        
    def save_catalogue(self, catalogue, name='initial'):
        """
        Save the catalogue made by `CatalogueConstructor` and needed
        by `Peeler` inside the working dir.
        
        Note that you can construct several catalogue for the same dataset
        to compare then just change the name. Different folder name so.
        
        """
        catalogue = dict(catalogue)
        chan_grp = catalogue['chan_grp']
        dir = os.path.join(self.dirname,'channel_group_{}'.format(chan_grp), 'catalogues', name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        arrays = ArrayCollection(parent=None, dirname=dir)
        
        to_rem = []
        for k, v in catalogue.items():
            if isinstance(v, np.ndarray):
                arrays.add_array(k, v, 'memmap')
                to_rem.append(k)
        
        for k in to_rem:
            catalogue.pop(k)
        
        # JSON is not possible for now because some key in catalogue are integer....
        # So bad....
        #~ with open(os.path.join(dir, 'catalogue.json'), 'w', encoding='utf8') as f:
            #~ json.dump(catalogue, f, indent=4)
        with open(os.path.join(dir, 'catalogue.pickle'), 'wb') as f:
            pickle.dump(catalogue, f)
        
    
    def load_catalogue(self,  name='initial', chan_grp=0):
        """
        Load the catalogue dict.
        """
        dir = os.path.join(self.dirname,'channel_group_{}'.format(chan_grp), 'catalogues', name)
        filename = os.path.join(dir, 'catalogue.pickle')
        #~ with open(os.path.join(dir, 'catalogue.json'), 'r', encoding='utf8') as f:
                #~ catalogue = json.load(f)
        if not os.path.exists(filename):
            return
        
        with open(filename, 'rb') as f:
            catalogue = pickle.load(f)

        
        arrays = ArrayCollection(parent=None, dirname=dir)
        arrays.load_all()
        for k in arrays.keys():
            catalogue[k] = np.array(arrays.get(k), copy=True)
        
        
        return catalogue
    
    def export_spikes(self, export_path=None,
                split_by_cluster=False,  use_cell_label=True, formats=None):
        """
        Export spikes to other format (csv, matlab, excel, ...)
        
        Parameters
        ------------------
        export_path: str or None
           export path. If None (default then inside working dir)
        split_by_cluster: bool (default False)
           Each cluster is split to a diffrent file or not.
        use_cell_label: bool (deafult True)
            if true cell_label is used if false cluster_label is used
        formats: 'csv' or 'mat' or 'xlsx'
            The output format.
        """
        
        if export_path is None:
            export_path = os.path.join(self.dirname, 'export')
        
        if formats is None:
            exporters = export_list
        elif isinstance(formats, str):
            assert formats in export_dict
            exporters = [ export_dict[formats] ]
        elif isinstance(format, list):
            exporters = [ export_dict[format] for format in formats]
        else:
            return
        
        for chan_grp in self.channel_groups.keys():
            
            catalogue = self.load_catalogue(chan_grp=chan_grp)
            if catalogue is None:
                continue
            
            if not self.is_spike_computed(chan_grp=chan_grp):
                continue
            
            for seg_num in range(self.nb_segment):
                spikes = self.get_spikes(seg_num=seg_num, chan_grp=chan_grp)
                
                if spikes is None: continue
                
                args = (spikes, catalogue, seg_num, chan_grp, export_path,)
                kargs = dict(split_by_cluster=split_by_cluster, use_cell_label=use_cell_label)
                for exporter in exporters:
                    exporter(*args, **kargs)
    
