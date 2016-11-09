import os
import json
import numpy as np


class ArrayCollection:
    """
    Collection of arrays.
    Some live in ram some ondisk via memmap.
    Some can be appendable.
    Automatique seattr to parent.
    """
    def __init__(self, parent=None, dirname=None):
        assert parent is not None
        
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
                d[name] = dict(dtype=self._array[name].dtype.name, shape=list(self._array[name].shape))
            json.dump(d, f, indent=4)        
        
    def create_array(self, name, dtype, shape, memory_mode):
        if memory_mode=='ram':
            arr = np.zeros(shape, dtype=dtype)
        elif memory_mode=='memmap':
            arr = np.memmap(self._fname(name), dtype=dtype, mode='w+', shape=shape)
        self._array[name] = arr
        self._array_attr[name] = {'state':'w', 'memory_mode':memory_mode}
        
        setattr(self.parent, name, self._array[name])
        self.flush_json()
        return arr
    
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
            self._array[name] = open(self._fname(name), mode='wb')
        
        self._array_attr[name] = {'state':'a', 'memory_mode':memory_mode, 'dtype': dtype, 'shape':shape}
        
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
        
        setattr(self.parent, name, self._array[name])
        self.flush_json()
    
    def load_if_exists(self, name):
        try:
            with open(self._fname('arrays', ext='.json'), 'r', encoding='utf8') as f:
                d = json.load(f)
                if name in d:
                    self._array[name] = np.memmap(self._fname(name), dtype=d[name]['dtype'], mode='r+').reshape(d[name]['shape'])
                    self._array_attr[name] = {'state':'r', 'memory_mode':'memmap'}
                    setattr(self.parent, name, self._array[name])
                else:
                    setattr(self.parent, name, None)
        except:
            setattr(self.parent, name, None)
    