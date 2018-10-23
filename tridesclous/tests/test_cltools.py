import pytest

from tridesclous.cltools import HAVE_PYOPENCL, get_cl_device_list

if HAVE_PYOPENCL:
    import pyopencl


@pytest.mark.skipif(not HAVE_PYOPENCL, reason='need OpenCL')
def test_get_cl_device_list():
        
        device_indexes = get_cl_device_list()
        print(device_indexes)





if __name__ == '__main__':
    test_get_cl_device_list()
    
    
    