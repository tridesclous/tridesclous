import pytest

from tridesclous.tests.testingtools import ON_CI_CLOUD
from tridesclous.gui.tests.testingguitools import HAVE_QT5


if  HAVE_QT5:
    import  pyqtgraph as pg
    from tridesclous.gui import *
    from tridesclous.gui.tools import *
    from tridesclous.gui.gui_params import cluster_params_by_methods, fullchain_params



from pprint import pprint

@pytest.mark.skip()
def test_open_dialog_catalogue():
    app = pg.mkQApp()
    dialog = ParamDialog(fullchain_params)
    dialog.resize(200,600)
    dialog.exec_()
    d = dialog.get()
    
    pprint(d)
    


@pytest.mark.skip()
def test_open_dialog_methods():
    app = pg.mkQApp()
    #~ method, kargs = open_dialog_methods(cluster_params_by_methods, None, title='Which method ?', selected_method=None)
    #~ method, kargs = open_dialog_methods(cluster_params_by_methods, None, title='Which method ?', selected_method='gmm')
    
    method, kargs = open_dialog_methods(cluster_params_by_methods, None, title='Which method ?')
    
    #~ open_dialog_methods.exec_()
    print(method, kargs)
    


if __name__ == '__main__':
    test_open_dialog_catalogue()
    
    #~ test_open_dialog_methods()
