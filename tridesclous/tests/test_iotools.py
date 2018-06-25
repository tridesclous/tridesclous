import numpy as np

import shutil, os

from tridesclous.iotools import ArrayCollection
import pytest

    

def one_test_ArrayCollection(withparent=False):
    #this is bugguy on windows
    
    print('test_ArrayCollection withparent=', withparent)
    if os.path.exists('test_ArrayCollection'):
        shutil.rmtree('test_ArrayCollection')
    
    if withparent:
        class Parent(object):
            pass
        parent = Parent()
    else:
        parent = None
    
    ac = ArrayCollection(dirname='test_ArrayCollection', parent=parent)
    
    data = np.zeros((8), dtype='float32')
    ac.add_array('data', data, 'memmap')
    
    assert ac.get('data').size==8
    
    # raise error if replace array when already referenced
    data2 = np.ones((3), dtype='float32')
    other_ref = ac.get('data')
    with pytest.raises(AssertionError):
        ac.add_array('data', data2, 'memmap')
    
    # other ref is copied so now able to replace
    other_ref = other_ref.copy()
    ac.add_array('data', data2, 'memmap')
    
    
    
    assert ac.get('data').size==data2.size
    assert ac.get('data').dtype==data2.dtype
    
    
    # more trick make a ref and do a reload
    # this make a ref on the old memap and this not detectable
    other_ref = ac.get('data')
    ac.load_if_exists('data')
    
    data6 = np.ones((5), dtype='float32')
    # this is the case with WARNING open r+
    ac.add_array('data', data6, 'memmap')
    
    del other_ref
    empty_arr = np.empty((0,0,0), dtype='float32')
    ac.add_array('data', empty_arr, 'memmap')
    empty_arr = ac.get('data')
    assert empty_arr.size == 0
    del empty_arr
    
    data7 = np.ones((0), dtype='float32')
    ac.add_array('data', data7, 'memmap')

def test_ArrayCollection():
    one_test_ArrayCollection(withparent=False)
    one_test_ArrayCollection(withparent=True)
    
    
if __name__=='__main__':
    test_ArrayCollection()