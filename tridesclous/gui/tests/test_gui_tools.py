from tridesclous.gui.tools import *
from tridesclous.gui.gui_params import cluster_params_by_methods

import pytest

@pytest.mark.skip()
def test_open_dialog_methods():
    app = pg.mkQApp()
    #~ method, kargs = open_dialog_methods(cluster_params_by_methods, None, title='Which method ?', selected_method=None)
    #~ method, kargs = open_dialog_methods(cluster_params_by_methods, None, title='Which method ?', selected_method='gmm')
    
    method, kargs = open_dialog_methods(cluster_params_by_methods, None, title='Which method ?')
    
    #~ open_dialog_methods.exec_()
    print(method, kargs)
    


if __name__ == '__main__':
    test_open_dialog_methods()
