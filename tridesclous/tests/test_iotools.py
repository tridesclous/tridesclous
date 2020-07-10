import numpy as np

import shutil, os

from tridesclous.iotools import ArrayCollection
import pytest

    

def one_test_ArrayCollection(withparent=False, memory_mode='memmap'):
    #this is bugguy on windows
    print()
    print('test_ArrayCollection withparent=', withparent, 'memory_mode=', memory_mode)
    if memory_mode == 'memmap' and os.path.exists('test_ArrayCollection'):
        shutil.rmtree('test_ArrayCollection')
    
    if withparent:
        class Parent(object):
            pass
        parent = Parent()
    else:
        parent = None
    
    ac = ArrayCollection(dirname='test_ArrayCollection', parent=parent)
    
    # make an array
    data = np.zeros((8), dtype='float32')
    ac.add_array('data', data, memory_mode)
    assert ac.get('data').size==8

    # make bigger array
    data = np.zeros((16), dtype='float32')
    ac.add_array('data', data, memory_mode)
    assert ac.get('data').size==16
    
    # make smaller array
    data = np.zeros((4), dtype='float32')
    ac.add_array('data', data, memory_mode)
    assert ac.get('data').size==4
    
    # append_array
    ac.append_array('data', data)
    assert ac.get('data').size==8
    
    
    if memory_mode == 'memmap':
        # raise error if replace array when already referenced
        data2 = np.ones((3), dtype='float32')
        other_ref = ac.get('data')
        with pytest.raises(AssertionError):
            ac.add_array('data', data2, memory_mode)
    
        # other ref is copied so now able to replace
        other_ref = other_ref.copy()
        ac.add_array('data', data2, memory_mode)
    
        assert ac.get('data').size==data2.size
        assert ac.get('data').dtype==data2.dtype
    
    
        # more tricky case : make a ref and do a reload
        # this make a ref on the old memap and this not detectable
        other_ref = ac.get('data')
        ac.load_if_exists('data')
        
        data6 = np.ones((5), dtype='float32') * 5
        # this is the case with WARNING open r+
        ac.add_array('data', data6, memory_mode)
        #~ print(ac.get('data'))
        #~ print(other_ref)
    
        del other_ref
    
    empty_arr = np.empty((0,0,0), dtype='float32')
    ac.add_array('data', empty_arr, memory_mode)
    empty_arr = ac.get('data')
    assert empty_arr.size == 0
    del empty_arr
    
    
    # special case empty array
    data7 = np.ones((0), dtype='float32')
    ac.add_array('data', data7, memory_mode)


    # special dtype
    data_field = np.zeros((5), dtype=[('a', 'int64'), ('b', 'S12')])
    ac.add_array('data_field', data_field, memory_mode)

    
    # test appendable array: 2 times
    for _ in range(2):
        ac.initialize_array('data_append', memory_mode, 'float32', (-1, 5))
        for i in range(3):
            arr_chunk = np.ones((1,5), dtype='float32')*i
            ac.append_chunk('data_append', arr_chunk)
        ac.finalize_array('data_append')
        assert ac.get('data_append').shape == (3,5)
    
    # annotations
    ac.annotate('data', one_annotation='yep')
    value = ac.get_annotation('data', 'one_annotation')
    assert value == 'yep'
    
    
    
    
    

def test_ArrayCollection():
    one_test_ArrayCollection(withparent=False, memory_mode='memmap')
    one_test_ArrayCollection(withparent=True, memory_mode='memmap')
    one_test_ArrayCollection(withparent=False, memory_mode='ram')
    one_test_ArrayCollection(withparent=True, memory_mode='ram')
    
    
if __name__=='__main__':
    test_ArrayCollection()
