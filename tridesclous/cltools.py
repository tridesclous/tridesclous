
from collections import OrderedDict

try:
    import pyopencl
    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False
    
    
    
class OpenCL_Helper:

    def initialize_opencl(self, cl_platform_index=None, cl_device_index=None):
        assert HAVE_PYOPENCL, 'PyOpenCL is not installed'
        
        if cl_platform_index is None:
            self.ctx = pyopencl.create_some_context(interactive=False)
        else:
            self.cl_platform_index = cl_platform_index
            self.cl_device_index = cl_device_index
            self.devices = [pyopencl.get_platforms()[cl_platform_index].get_devices()[cl_device_index]]
            self.ctx = pyopencl.Context(self.devices)
        
        self.queue = pyopencl.CommandQueue(self.ctx)
        
        #~ print(self.ctx)



def get_cl_device_list():
    
    device_indexes = []
    for platform_index, platform in enumerate(pyopencl.get_platforms()):
        for device_index,device in enumerate(platform.get_devices()):
            device_indexes.append((device.name, platform_index, device_index))
    
    return device_indexes
