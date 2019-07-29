
from collections import OrderedDict

try:
    import pyopencl
    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False
    

default_platform_index = None
default_device_index = None



class OpenCL_Helper:
    """
    Used in Peeler and SignalPreprocessor_OpenCL.
    
    Make possible to select device/platform with 3 flavor:
       * default at tridesclous level when default_platform_index/default_device_index is not None
       * at each Peeler.set_params(cl_platform_index=XXX, cl_device_index=YYY) to change this for each instance (make compraison)
       * default at pyopencl level when default_platform_index is None
    
    """

    def initialize_opencl(self, cl_platform_index=None, cl_device_index=None, ctx=None, queue=None):
        assert HAVE_PYOPENCL, 'PyOpenCL is not installed'

        global default_platform_index
        global default_device_index
        
        if ctx is None and queue is None:
            if cl_platform_index is None:
                if default_platform_index is not None and default_device_index is not None:
                    self.cl_platform_index = default_platform_index
                    self.cl_device_index = default_device_index
                    self.devices = [pyopencl.get_platforms()[self.cl_platform_index].get_devices()[self.cl_device_index]]
                    self.ctx = pyopencl.Context(self.devices)
                else:
                    self.ctx = pyopencl.create_some_context(interactive=False)
            else:
                self.cl_platform_index = cl_platform_index
                self.cl_device_index = cl_device_index
                self.devices = [pyopencl.get_platforms()[self.cl_platform_index].get_devices()[self.cl_device_index]]
                self.ctx = pyopencl.Context(self.devices)
            self.queue = pyopencl.CommandQueue(self.ctx)
        else:
            assert cl_platform_index is None and cl_device_index is None
            self.ctx = ctx
            self.queue = queue
        
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        


def get_cl_device_list():
    
    device_indexes = []
    for platform_index, platform in enumerate(pyopencl.get_platforms()):
        for device_index,device in enumerate(platform.get_devices()):
            device_indexes.append((device.name, platform_index, device_index))
    
    return device_indexes


def set_default_cl_device(platform_index=None, device_index=None):
    global default_platform_index
    global default_device_index
    
    default_platform_index = platform_index
    default_device_index = device_index


