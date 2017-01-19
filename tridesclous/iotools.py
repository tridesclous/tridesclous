import os
import json
import numpy as np
import io
import sys

class ArrayCollection:
    """
    Collection of arrays.
    Some live in ram some ondisk via memmap.
    Some can be appendable.
    Automatique seattr to parent.
    """
    def __init__(self, parent=None, dirname=None):
        #~ assert parent is not None
        
        self.dirname = dirname
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        
        self.parent = parent
        self._array = {}
        self._array_attr = {}

    def _fname(self, name, ext='.raw'):
        filename = os.path.join(self.dirname, name+ext)
        return filename

    def flush_json(self):
        with open(self._fname('arrays', ext='.json'), 'w', encoding='utf8') as f:
            d = {}
            for name in self._array:
                if self._array_attr[name]['state']=='a':
                    continue
                if self._array[name].dtype.fields is None:
                    dt = self._array[name].dtype.name
                else:
                    dt = self._array[name].dtype.descr
                d[name] = dict(dtype=dt, shape=list(self._array[name].shape))
            json.dump(d, f, indent=4)        
    
    def _fix_existing(self, name):
        # deal with a bug on windows when creating a memmap in w+
        # when the file already existing in r+ mode
        #~ print(sys.platform)
        if not sys.platform.startswith('win'):
            return 'w+'
        if name in self._array:
            #~ print('create_array oups exists!!!', name)
            #~ print(type(self._array[name]))
            #~ print('new array', dtype, shape)
            if isinstance(self._array[name], np.memmap):
                a = self._array.pop(name)
                #~ print('old array', a.dtype, a.shape, a.mode)#, a.filename)
                a._mmap.close()
                if self.parent is not None:
                    delattr(self.parent, name)
                del(a)
            elif isinstance(self._array[name], io.IOBase):
                a = self._array.pop(name)
                a.close()
            #~ if os.path.exists(self._fname(name)):
                #~ print('remove', self._fname(name))
                #~ os.remove(self._fname(name))
            mode='r+'
        else:
            mode='w+'
        #~ print('mode', mode)
        return mode
    
    def create_array(self, name, dtype, shape, memory_mode):
        
        if memory_mode=='ram':
            arr = np.zeros(shape, dtype=dtype)
        elif memory_mode=='memmap':
            mode = self._fix_existing(name)
            #~ arr = np.memmap(self._fname(name), dtype=dtype, mode='w+', shape=shape)
            arr = np.memmap(self._fname(name), dtype=dtype, mode=mode, shape=shape)
        self._array[name] = arr
        self._array_attr[name] = {'state':'w', 'memory_mode':memory_mode}
        
        if self.parent is not None:
            setattr(self.parent, name, self._array[name])
        self.flush_json()
        return arr
    
    def add_array(self, name, data, memory_mode):
        self.create_array(name, data.dtype, data.shape, memory_mode)
        self._array[name][:] = data

    
    def delete_array(self, name):
        self.detach_array(name)
        raise(NotImplementedError)
        
    def detach_array(self, name):
        self._array.pop(name)
        self._array_attr.pop(name)
        delattr(self.parent, name)
        self.flush_json()
    
    def initialize_array(self, name, memory_mode, dtype, shape):
        if memory_mode=='ram':
            self._array[name] = []
        elif memory_mode=='memmap':
            mode = self._fix_existing(name)
            self._array[name] = open(self._fname(name), mode='wb+')
        
        self._array_attr[name] = {'state':'a', 'memory_mode':memory_mode, 'dtype': dtype, 'shape':shape}
        
        if self.parent is not None:
            setattr(self.parent, name, None)
    
    def append_chunk(self, name, arr_chunk):
        assert self._array_attr[name]['state']=='a'
        
        memory_mode = self._array_attr[name]['memory_mode']
        if memory_mode=='ram':
            self._array[name].append(arr_chunk)
        elif memory_mode=='memmap':
            self._array[name].write(arr_chunk.tobytes(order='C'))
        
        
    def finalize_array(self, name):
        assert self._array_attr[name]['state']=='a'
        
        memory_mode = self._array_attr[name]['memory_mode']
        if memory_mode=='ram':
            self._array[name] = np.concatenate(self._array[name], axis=0)
        elif memory_mode=='memmap':
            self._array[name].close()
            self._array[name] = np.memmap(self._fname(name), dtype=self._array_attr[name]['dtype'],
                                                mode='r+').reshape(self._array_attr[name]['shape'])
        
        self._array_attr[name]['state'] = 'r'
        if self.parent is not None:
            setattr(self.parent, name, self._array[name])
        self.flush_json()
    
    def flush_array(self, name):
        memory_mode = self._array_attr[name]['memory_mode']
        if memory_mode=='ram':
            pass
        elif memory_mode=='memmap':
            self._array[name].flush()
    
    
    def load_if_exists(self, name):
        try:
            with open(self._fname('arrays', ext='.json'), 'r', encoding='utf8') as f:
                d = json.load(f)
                if name in d:
                    if isinstance(d[name]['dtype'], str):
                        dtype = np.dtype(d[name]['dtype'])
                    else:
                        dtype = np.dtype([ (k,v) for k,v in d[name]['dtype']])
                    #TODO fix this
                    arr = np.memmap(self._fname(name), dtype=dtype, mode='r+')
                    #~ print(arr.shape, np.prod(d[name]['shape']))
                    arr = arr[:np.prod(d[name]['shape'])]
                    arr = arr.reshape(d[name]['shape'])
                    self._array[name] = arr
                    self._array_attr[name] = {'state':'r', 'memory_mode':'memmap'}
                    if self.parent is not None:
                        setattr(self.parent, name, self._array[name])
                else:
                    if self.parent is not None:
                        setattr(self.parent, name, None)
        except:
            #~ print('erreur load', name)
            if self.parent is not None:
                setattr(self.parent, name, None)
    
    def load_all(self):
        with open(self._fname('arrays', ext='.json'), 'r', encoding='utf8') as f:
            d = json.load(f)
            all_keys = list(d.keys())
        for k in all_keys:
            self.load_if_exists(k)
    
    def get(self, name):
        assert name in self._array_attr
        return self._array[name]
    
    def keys(self):
        return self._array.keys()
    
    
