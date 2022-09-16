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
import packaging.version

import sklearn.metrics

from .version import version as tridesclous_version
from .datasource import data_source_classes
from .iotools import ArrayCollection
from .tools import download_probe, create_prb_file_from_dict, fix_prb_file_py2
from .waveformtools import extract_chunks

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
            self.info['tridesclous_version'] = tridesclous_version
            self.flush_info()
            self.datasource = None
        else:
            with open(self.info_filename, 'r', encoding='utf8') as f:
                self.info = json.load(f)
            
            self._check_tridesclous_version()
            if len(self.info)>1:
                self._reload_channel_group()
                self._reload_data_source()
                self._reload_data_source_info()
                self._open_processed_data()
            else:
                self.datasource = None
                
            #~ except:
                #~ self.info = {}
                #~ self.flush_info()
                #~ self.datasource = None
 
    def __repr__(self):
        t = "DataIO <id: {}> \n  workdir: {}\n".format(id(self), self.dirname)
        if len(self.info) <= 1 and self.datasource is None:
            t  += "  Not datasource set yet"
            return t
        t += "  sample_rate: {}\n".format(self.sample_rate)
        t += "  total_channel: {}\n".format(self.total_channel)
        if len(self.channel_groups)==1:
            k0, cg0 = next(iter(self.channel_groups.items()))
            ch_names = np.array(self.all_channel_names)[cg0['channels']]
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
            lengths = [ self.segment_shapes[i][0] for i in range(self.nb_segment)]
            t += '  length: '+' '.join('{}'.format(l) for l in lengths)+'\n'
            t += '  durations: '+' '.join('{:0.1f}'.format(l/self.sample_rate) for l in lengths)+' s.\n'
        if t.endswith('\n'):
            t = t[:-1]
        
        return t

 
    def flush_info(self):
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f, indent=4)
    
    def _check_tridesclous_version(self):
        folder_version= self.info.get('tridesclous_version', 'unknown')
        
        if folder_version == 'unknown':
            w = True
        else:
            v1 = packaging.version.parse(tridesclous_version)
            v2 = packaging.version.parse(self.info['tridesclous_version'])
            if (v1.major == v2.major) and (v1.minor == v2.minor):
                w = False
            else:
                w = True

        if w:
            txt = 'This folder was created with an old tridesclous version ({})\n'\
                    'The actual version is {}\n'\
                    'You may have bug in internal structure.'
            print(txt.format(folder_version, tridesclous_version))
    
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
        elif 'dirnames' in kargs:
            kargs['dirnames'] = [ os.path.abspath(f) for f in kargs['dirnames']]
        else:
            raise ValueError('set_data_source() must contain filenames or dirnames')

        self.info['datasource_type'] = type
        self.info['datasource_kargs'] = kargs
        self._reload_data_source()
        
        # be default chennel group all channels
        channel_groups = {0:{'channels':list(range(self.total_channel))}}
        self.set_channel_groups( channel_groups, probe_filename='default.prb')

        self.flush_info()
        
        self._reload_data_source_info()
        
        # this create segment path
        self._open_processed_data()
    
    def _reload_data_source(self):
        assert 'datasource_type' in self.info
        kargs = self.info['datasource_kargs']
        try:
            self.datasource = data_source_classes[self.info['datasource_type']](**kargs)
        except:
            print('The datatsource is not found', self.info['datasource_kargs'])
            self.datasource = None
        self._reload_data_source_info()
    
    def _save_datasource_info(self):
        assert self.datasource is not None, 'Impossible to load datasource and get info'
        # put some info of datasource
        
        nb_seg = self.datasource.nb_segment
        clean_shape = lambda  shape: tuple(int(e) for e in shape)
        segment_shapes = [clean_shape(self.datasource.get_segment_shape(s)) for s in range(nb_seg)]
        self.info['datasource_info'] = dict(
            total_channel=int(self.datasource.total_channel),
            nb_segment=int(nb_seg),
            sample_rate=float(self.datasource.sample_rate),
            source_dtype=str(self.datasource.dtype),
            all_channel_names=[str(name) for name in self.datasource.get_channel_names()],
            segment_shapes = segment_shapes,
        )
        self.flush_info()

    
    def _reload_data_source_info(self):
        if 'datasource_info' in self.info:
            # no need for datasource
            d = self.info['datasource_info'] 
            self.total_channel = d['total_channel']
            self.nb_segment = d['nb_segment']
            self.sample_rate = d['sample_rate']
            self.source_dtype = np.dtype(d['source_dtype'])
            self.all_channel_names =  d['all_channel_names']
            self.segment_shapes = d['segment_shapes']
        else:
            # This cas is for old directories were
            self._save_datasource_info()
            self._reload_data_source_info()
   
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
            # assume that it is a linear probes with 100 um
            geometry = { c: [0, i*100] for i, c in enumerate(channels) }
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
    
    def get_channel_distances(self, chan_grp=0):
        geometry = self.get_geometry(chan_grp=chan_grp)
        distances = sklearn.metrics.pairwise.euclidean_distances(geometry)
        return distances
    
    def get_channel_adjacency(self, chan_grp=0, adjacency_radius_um=None):
        assert adjacency_radius_um is not None
        channel_distances = self.get_channel_distances(chan_grp=chan_grp)
        channels_adjacency = {}
        nb_chan = self.nb_channel(chan_grp=chan_grp)
        for c in range(nb_chan):
            nearest, = np.nonzero(channel_distances[c, :] < adjacency_radius_um)
            channels_adjacency[c] = nearest
        return channels_adjacency

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
        ch_names = np.array(self.all_channel_names)[channels]
        
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
        full_shape =  self.segment_shapes[seg_num]
        return full_shape[0]
    
    def get_segment_shape(self, seg_num, chan_grp=0):
        """
        Segment shape for a given segment index and channel group.
        """
        full_shape =  self.segment_shapes[seg_num]
        shape = (full_shape[0], self.nb_channel(chan_grp))
        return shape
    
    def get_duration_per_segments(self, total_duration=None):
        duration_per_segment = []
        
        if total_duration is not None:
            remain = float(total_duration)
        
        for seg_num in range(self.nb_segment):
            dur = self.get_segment_length(seg_num=seg_num) / self.sample_rate
            
            if total_duration is None:
                duration_per_segment.append(dur)
            elif remain ==0:
                duration_per_segment.append(0.)
            elif dur <=remain:
                duration_per_segment.append(dur)
                remain -= dur
            else:
                duration_per_segment.append(remain)
                remain = 0.
        
        return duration_per_segment
    
    def get_signals_chunk(self, seg_num=0, chan_grp=0,
                i_start=None, i_stop=None,
                signal_type='initial', pad_width=0):
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
        pad_width: int (0 default)
            Add optional pad on each sides
            usefull for filtering border effect
        
        """
        channels = self.channel_groups[chan_grp]['channels']
        
        after_padding = False
        after_padding_left = 0
        after_padding_right = 0
        if pad_width > 0:
            i_start = i_start - pad_width
            i_stop =  i_stop + pad_width
        
            if i_start < 0:
                after_padding = True
                after_padding_left = -i_start
                i_start = 0
            if i_stop > self.get_segment_length(seg_num):
                after_padding = True
                after_padding_right = i_stop - self.get_segment_length(seg_num)
                i_stop = self.get_segment_length(seg_num)
        
        if signal_type=='initial':
            data = self.datasource.get_signals_chunk(seg_num=seg_num, i_start=i_start, i_stop=i_stop)
            data = data[:, channels]
        elif signal_type=='processed':
            data = self.arrays[chan_grp][seg_num].get('processed_signals')[i_start:i_stop, :]
        else:
            raise(ValueError, 'signal_type is not valide')
        
        if after_padding:
            # finalize padding on border
            data2 = np.zeros((data.shape[0] + after_padding_left + after_padding_right, data.shape[1]), dtype=data.dtype)
            data2[after_padding_left:data2.shape[0]-after_padding_right, :] = data
            data = data2
        
        return data
    
    def iter_over_chunk(self, seg_num=0, chan_grp=0,  i_stop=None,
                        chunksize=1024, pad_width=0, with_last_chunk=False,   **kargs):
        """
        Create an iterable on signals. ('initial' or 'processed')
        
        Usage
        ----------
        
            for ind, sig_chunk in data.iter_over_chunk(seg_num=0, chan_grp=0, chunksize=1024, signal_type='processed'):
                do_something_on_chunk(sig_chunk)
        
        """
        seg_length = self.get_segment_length(seg_num)
        
        length = seg_length
        if i_stop is not None:
            length = min(length, i_stop)
        
        total_length = length + pad_width
        
        nloop = total_length//chunksize
        if total_length % chunksize and with_last_chunk:
            nloop += 1
        
        last_sample = None
        for i in range(nloop):
            i_stop = (i+1)*chunksize
            i_start = i_stop - chunksize
            
            if i_stop > seg_length:
                sigs_chunk2 = np.zeros((chunksize, sigs_chunk.shape[1]), dtype=sigs_chunk.dtype)
                if i_start < seg_length:
                    sigs_chunk = self.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp, i_start=i_start, i_stop=seg_length, **kargs)
                    sigs_chunk2[:sigs_chunk.shape[0], :] = sigs_chunk
                    last_sample = sigs_chunk[-1, :]
                    # extend with last sample : agttenuate fileter border effect
                    sigs_chunk2[sigs_chunk.shape[0]:, :] = last_sample
                else:
                    if last_sample is not None:
                        sigs_chunk2[:, :] = last_sample
                yield  i_stop, sigs_chunk2
                
            else:
                sigs_chunk = self.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp, i_start=i_start, i_stop=i_stop, **kargs)
                if i_stop == seg_length:
                    last_sample = sigs_chunk[-1, :]
                yield  i_stop, sigs_chunk
        
        #~ if with_last_chunk and i_stop<total_length:
            #~ i_start = i_stop
            #~ i_stop = length
            #~ sigs_chunk = self.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp, i_start=i_start, i_stop=i_stop, **kargs)
            
            #~ sigs_chunk2 = np.zeros((chunksize, sigs_chunk.shape[1]), dtype=sigs_chunk.dtype)
            #~ if sigs_chunk.shape[0] > 0:
                #~ sigs_chunk2[:sigs_chunk.shape[0], :] = sigs_chunk
                #~ # extend with last sample : agttenuate fileter border effect
                #~ sigs_chunk2[sigs_chunk.shape[0]:, :] = sigs_chunk[-1, :]
            
            #~ yield  i_start+chunksize, sigs_chunk2
    
    def reset_processed_signals(self, seg_num=0, chan_grp=0, dtype='float32'):
        """
        Reset processed signals.
        """
        self.arrays[chan_grp][seg_num].create_array('processed_signals', dtype, 
                            self.get_segment_shape(seg_num, chan_grp=chan_grp), 'memmap')
        self.arrays[chan_grp][seg_num].annotate('processed_signals', processed_length=0)
    
    def set_signals_chunk(self,sigs_chunk, seg_num=0, chan_grp=0, i_start=None, i_stop=None, signal_type='processed'):
        """
        Set a signal chunk (only for 'processed')
        """
        assert signal_type != 'initial'

        if signal_type=='processed':
            data = self.arrays[chan_grp][seg_num].get('processed_signals')
            data[i_start:i_stop, :] = sigs_chunk
        
    def flush_processed_signals(self, seg_num=0, chan_grp=0, processed_length=-1):
        """
        Flush the underlying memmap for processed signals.
        """
        self.arrays[chan_grp][seg_num].flush_array('processed_signals')
        self.arrays[chan_grp][seg_num].annotate('processed_signals', processed_length=processed_length)
    
    def get_processed_length(self, seg_num=0, chan_grp=0):
        """
        Get the length in sample how already processed part of the segment.
        """
        return self.arrays[chan_grp][seg_num].get_annotation('processed_signals', 'processed_length')
    
    def already_processed(self, seg_num=0, chan_grp=0, length=None):
        """
        Check if the segment is entirely processedis already computed until length
        """
        # check if signals are processed
        if length is None:
            length = self.get_segment_length(seg_num)
        already_done = self.get_processed_length(seg_num, chan_grp=chan_grp)
        return  already_done >= length
    
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

    def get_peak_values(self,  seg_num=0, chan_grp=0, sample_indexes=None, channel_indexes=None):
        """
        Extract peak values
        """
        assert sample_indexes is not None, 'Provide sample_indexes'
        assert channel_indexes is not None, 'Provide channel_indexes'
        sigs = self.arrays[chan_grp][seg_num].get('processed_signals')
        
        peak_values = []
        for s, c in zip(sample_indexes, channel_indexes):
            peak_values.append(sigs[s, c])
        peak_values = np.array(peak_values)
        return peak_values
        
    
    def get_some_waveforms(self, seg_num=None, seg_nums=None, chan_grp=0, peak_sample_indexes=None,
                                n_left=None, n_right=None, waveforms=None, channel_indexes=None):
        """
        Exctract some waveforms given sample_indexes
        
        seg_num is int then all spikes come from same segment
        
        if seg_num is None then seg_nums is an array that contain seg_num for each spike.
        """
        assert peak_sample_indexes is not None, 'Provide sample_indexes'
        peak_width = n_right - n_left
        
        if channel_indexes is None:
            nb_chan = self.nb_channel(chan_grp)
        else:
            nb_chan = len(channel_indexes)
        
        if waveforms is None:
            dtype = self.arrays[chan_grp][0].get('processed_signals').dtype
            waveforms = np.zeros((peak_sample_indexes.size, peak_width, nb_chan), dtype=dtype)
        else:
            assert waveforms.shape[0] == peak_sample_indexes.size
            assert waveforms.shape[1] == peak_width
        
        if seg_num is not None:
            left_indexes = peak_sample_indexes + n_left
            sigs = self.arrays[chan_grp][seg_num].get('processed_signals')
            extract_chunks(sigs, left_indexes, peak_width, 
                                        channel_indexes=channel_indexes, chunks=waveforms)
        elif seg_nums is not None and isinstance(seg_nums, np.ndarray):
            n = 0
            for seg_num in np.unique(seg_nums):
                
                mask = seg_num == seg_nums
                left_indexes = peak_sample_indexes[mask]+n_left
                if left_indexes.size == 0:
                    continue
                chunks = waveforms[n:n+left_indexes.size] # this avoid a copy
                sigs = self.arrays[chan_grp][seg_num].get('processed_signals')
                extract_chunks(sigs, left_indexes, peak_width, 
                                            channel_indexes=channel_indexes, chunks=chunks)
                n += left_indexes.size
        else:
            raise 'error seg_num or seg_nums'
        
        return waveforms


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
    
    def get_log_path(self, chan_grp=0):
        cg_path = os.path.join(self.dirname, 'channel_group_{}'.format(chan_grp))
        log_path = os.path.join(cg_path, 'log')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        return log_path
