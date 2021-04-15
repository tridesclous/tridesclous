import os
import json
import io
import sys
import shutil
import gc

import numpy as np

class ArrayCollection:
    """
    Collection of arrays.
    Some live in ram some ondisk via memmap.
    Some can be appendable.
    Automatique seattr to parent.
    """
    def __init__(self, parent=None, dirname=None):
        #~ assert parent is not None
        
        self.dirname = os.path.abspath(dirname)
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
                d[name] = dict(dtype=dt,
                                shape=list(self._array[name].shape),
                                annotations=self._array_attr[name]['annotations']
                                )
                
            json.dump(d, f, indent=4)        
    
    def _nb_ref(self, name):
        nb_ref = sys.getrefcount(self._array[name]) -1 
        return nb_ref
    
    def _check_nb_ref(self, name):
        """Check if an array is not refrenced outside this class and aparent
        Usefull before deleting.
        """
        # ok for py36
        # nb_ref = len(gc.get_referrers(self._array[name]))
        # ok for py36 and py37
        nb_ref = sys.getrefcount(self._array[name]) -1 
        
        if self.parent is None:
            nb_expect = 1
        else:
            nb_expect = 2
        assert nb_ref == nb_expect, '{} array is referenced outside {}!={}'.format(name, nb_ref, nb_expect)
        
    
    def _fix_existing(self, name):
        """
        Test if name is already in _array.
        If state='a', then finalize and detach.
        else detach.
        
        Then _check_nb_ref
        
        """
        
        if name in self._array and (self._array_attr[name]['memory_mode'] == 'memmap'):
            if (self._array_attr[name]['state'] == 'a'):
                # case appendable and memmap
                self._array[name].close()
                self._array.pop(name)
                self._array_attr.pop(name)
            else:
                # case memmap standard
                a = self._array.pop(name)
                if a.size>0:
                    # hack when size is 0
                    a._mmap.close()
                    if self.parent is not None:
                        delattr(self.parent, name)
                del(a)
                self._array_attr.pop(name)
                
        
        if os.path.exists(self._fname(name)):
            mode = 'r+'
        else:
            mode = 'w+'
        
        return mode
        
        #~ # deal with a bug on windows when creating a memmap in w+
        #~ # when the file already existing in r+ mode
        #~ if not sys.platform.startswith('win'):
            #~ return 'w+'
        #~ if name in self._array:
            #~ if isinstance(self._array[name], np.memmap):
                #~ a = self._array.pop(name)
                #~ a._mmap.close()
                #~ if self.parent is not None:
                    #~ delattr(self.parent, name)
                #~ del(a)
            #~ elif isinstance(self._array[name], io.IOBase):
                #~ a = self._array.pop(name)
                #~ a.close()
            
            #~ try:
                #~ if os.path.exists(self._fname(name)):
                    #~ os.remove(self._fname(name))
                #~ mode='w+'
            #~ except:
                #~ print('WARNING open r+')
                #~ mode='r+'

        #~ else:
            #~ mode='w+'
        #~ return mode
    
    def create_array(self, name, dtype, shape, memory_mode):
        
        if memory_mode=='ram':
            arr = np.zeros(shape, dtype=dtype)
        elif memory_mode=='memmap':
            
            if name in self._array:
                self._check_nb_ref(name)
            
            mode = self._fix_existing(name)
            # detect when 0 size because np.memmap  bug with size=0
            if np.prod(shape)!=0:
                arr = np.memmap(self._fname(name), dtype=dtype, mode=mode, shape=shape)
            else:
                with open(self._fname(name), mode=mode) as f:
                    f.write('')
                arr = np.empty(shape, dtype=dtype)
                #~ print('empty array memmap !!!!', name, shape)
        
        self._array[name] = arr
        self._array_attr[name] = {'state':'w', 'memory_mode':memory_mode, 'annotations':{}}
        
        if self.parent is not None:
            setattr(self.parent, name, self._array[name])
        self.flush_json()
        return arr
    
    def add_array(self, name, data, memory_mode):
        self.create_array(name, data.dtype, data.shape, memory_mode)
        self._array[name][:] = data
        self.flush_array(name)

    def delete_array(self, name):
        if name not in self._array:
            return
        
        if self._array_attr[name]['memory_mode'] == 'memmap':
            #delete file if exist
            filename = self._fname(name)
            os.remove(filename)
        self.detach_array(name)
        #~ raise(NotimplementedError)
        
        
    def detach_array(self, name, mmap_close=False):
        if name not in self._array:
            return
        a = self._array.pop(name)
        if mmap_close and hasattr(a, '_mmap'):
            a._mmap.close()
        self._array_attr.pop(name)
        if self.parent is not None:
            delattr(self.parent, name)
        self.flush_json()
    
    def initialize_array(self, name, memory_mode, dtype, shape):
        if memory_mode=='ram':
            self._array[name] = []
        elif memory_mode=='memmap':
            mode = self._fix_existing(name)
            self._array[name] = open(self._fname(name), mode='wb+')
        
        self._array_attr[name] = {'state':'a', 'memory_mode':memory_mode, 'dtype': dtype, 'shape':shape, 'annotations':{}}
        
        if self.parent is not None:
            setattr(self.parent, name, None)
        self.flush_json()
    
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
            try:
                self._array[name] = np.memmap(self._fname(name), dtype=self._array_attr[name]['dtype'],
                                                mode='r+').reshape(self._array_attr[name]['shape'])
            except ValueError:
                #empty file = 0 elements
                #FIXME: find something else
                self._array[name] = np.zeros((0,), dtype=self._array_attr[name]['dtype']).reshape(self._array_attr[name]['shape'])
            
        self._array_attr[name]['state'] = 'r'
        if self.parent is not None:
            setattr(self.parent, name, self._array[name])
        self.flush_json()
    
    def append_array(self, name, arr):
        # append array when not initialize_array (mode = 'a')
        assert self._array_attr[name]['state'] != 'a'
        #~ print('append_array', name, self._array[name].dtype, arr.dtype)
        assert self._array[name].dtype == arr.dtype

        memory_mode = self._array_attr[name]['memory_mode']
        if memory_mode=='ram':
            # self._array[name].append(arr)
            self._array[name] = np.append(self._array[name], arr)
        elif memory_mode=='memmap':
            old_shape = self._array[name].shape
            if len(old_shape) > 1:
                for i in range(1, len(old_shape)):
                    assert old_shape[i] == arr.shape[i] 
            new_shape = (old_shape[0] + arr.shape[0], ) + old_shape[1:]
            
            nbytes = self._array[name].nbytes
            # close memmap
            self._array[name] = None 
            #write
            with open(self._fname(name), mode='ab+') as f:
                f.seek(nbytes)
                f.write(arr.tobytes(order='C'))
            # reopen memmap
            self._array[name] = np.memmap(self._fname(name), dtype=arr.dtype,
                                                mode='r+', shape=new_shape)

            if self.parent is not None:
                setattr(self.parent, name, self._array[name])

    
    def flush_array(self, name):
        memory_mode = self._array_attr[name]['memory_mode']
        if memory_mode=='ram':
            pass
        elif memory_mode=='memmap':
            if self._array[name].size>0:
                self._array[name].flush()
    
    
    def load_if_exists(self, name):
        if not os.path.exists(self._fname('arrays', ext='.json')):
            if self.parent is not None:
                setattr(self.parent, name, None)
            return
        
        try:
            with open(self._fname('arrays', ext='.json'), 'r', encoding='utf8') as f:
                d = json.load(f)
        except:
            print('erreur load json', name)
            if self.parent is not None:
                setattr(self.parent, name, None)
            return
        
        if name in d:
            if os.path.exists(self._fname(name)):
                if isinstance(d[name]['dtype'], str):
                    dtype = np.dtype(d[name]['dtype'])
                else:
                    dtype = np.dtype([ (k,v) for k,v in d[name]['dtype']])
                shape = d[name]['shape']
                if np.prod(d[name]['shape'])>0:
                    arr = np.memmap(self._fname(name), dtype=dtype, mode='r+')
                    arr = arr[:np.prod(shape)]
                    arr = arr.reshape(shape)
                else:
                    # little hack array is empty
                    arr = np.empty(shape, dtype=dtype)
                self._array[name] = arr
                self._array_attr[name] = {'state':'r', 'memory_mode':'memmap'}
                self._array_attr[name]['annotations'] = d[name].get('annotations', {})
                if self.parent is not None:
                    setattr(self.parent, name, self._array[name])
            else:
                if self.parent is not None:
                    setattr(self.parent, name, self._array[name])
        else:
            if self.parent is not None:
                setattr(self.parent, name, None)
            
    #~ except:
        #~ print('erreur load', name)
        #~ if self.parent is not None:
            #~ setattr(self.parent, name, None)

        
    #~ except:
        #~ print('erreur load', name)
        #~ if self.parent is not None:
            #~ setattr(self.parent, name, None)
    
    def load_all(self):
        with open(self._fname('arrays', ext='.json'), 'r', encoding='utf8') as f:
            d = json.load(f)
            all_keys = list(d.keys())
        for k in all_keys:
            self.load_if_exists(k)
    
    def has_key(self, name):
        return name in self._array_attr
    
    def get(self, name):
        assert name in self._array_attr
        return self._array[name]
    
    def keys(self):
        return self._array.keys()
    
    def annotate(self, name, **kargs):
        assert name in self._array_attr
        self._array_attr[name]['annotations'].update(kargs)
        self.flush_json()
    
    def get_annotation(self, name, key):
        assert name in self._array_attr
        return self._array_attr[name]['annotations'][key]

