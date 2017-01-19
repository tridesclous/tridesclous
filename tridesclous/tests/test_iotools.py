import numpy as np

from tridesclous.iotools import ArrayCollection


def test_ArrayCollection():
    pass
    

def test_ArrayCollection_several_open():
    #this is bugguy on windows
    
    ac = ArrayCollection(dirname='test_ArrayCollection')
    data = np.zeros((400), dtype='float32')
    data2 = np.ones((3), dtype='float32')
    ac.add_array('data', data, 'memmap')
    
    data3 = ac.get('data')
    ac.add_array('data', data2, 'memmap')
    data4 = ac.get('data')

    print(type(data3), data3)
    print(type(data4), data4)
    
    ac.load_if_exists('data')
    data5 = ac.get('data')
    print(type(data5), data5)
    data6 = np.ones((5), dtype='float32')
    ac.add_array('data', data6, 'memmap')
    data7 = ac.get('data')
    print(type(data7), data7)
    
    
    
    
    
    
if __name__=='__main__':
    #~ test_ArrayCollection()
    test_ArrayCollection_several_open()