from .myqt import QT
import pyqtgraph as pg

from collections import OrderedDict

from .tools import ParamDialog
from ..cltools import get_cl_device_list, set_default_cl_device


    
class GpuSelector(ParamDialog):
    def __init__(self, settings=None, parent=None):
        self.settings = settings
        assert settings is not None
        
        
        
        title = 'Select default GPU for OpenCL task'
        
        self.possibles_gpu = OrderedDict()
        for name, platform_index, device_index in get_cl_device_list():
            self.possibles_gpu[name] = (platform_index, device_index)
        
        selected_cl_name = self.settings.value('selected_cl_name')
        #~ print('selected_cl_name', selected_cl_name)
        if selected_cl_name is None and len(self.possibles_gpu)>0:
            selected_cl_name = list(self.possibles_gpu.keys())[0]
            save = False
        elif selected_cl_name in self.possibles_gpu:
            save = True
        else:
            save = False
            
        
        l = list(self.possibles_gpu.keys())
        params = [
                {'name': 'always_use_default', 'type': 'bool', 'value': save},
                {'name': 'OpenCL_device', 'type': 'list', 'value' : selected_cl_name, 'limits': l},
        ]
        ParamDialog.__init__(self, params, title = '', parent=parent)

    def apply_cl_setting(self):
        """save and set default"""
        d = self.get()
        name = d['OpenCL_device']
        print('select', name)
        if d['always_use_default']:
            self.settings.setValue('selected_cl_name', name)
        else:
            self.settings.setValue('selected_cl_name', None)
        platform_index, device_index = self.possibles_gpu[name]
        set_default_cl_device(platform_index=None, device_index=None)

