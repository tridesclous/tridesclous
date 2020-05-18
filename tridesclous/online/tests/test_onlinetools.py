from pyacq.devices import OpenEphysGUIRelay
from tridesclous.online import start_online_window
import pyqtgraph as pg


def test_openephys_tridesclous():
    
    
    app = pg.mkQApp()
    
    dev = OpenEphysGUIRelay()
    dev.configure(openephys_url='tcp://127.0.0.1:20000')
    dev.outputs['signals'].configure()
    dev.initialize()
    
    prb_filename = 'probe_openephys_16ch.prb'
    
    man, win = start_online_window(dev, prb_filename, workdir=None, n_process=1, pyacq_manager=None)
    
    win.show()
    
    win.start()
    dev.start()
    
    app.exec_()



if __name__ == '__main__':
    test_openephys_tridesclous()
    
    #~ d = {}
    #~ d[0] = {}
    #~ d[0]['channels'] = list(range(16))
    #~ d[0]['geometry'] = {c: [0, c*50] for c in range(16) }
    
    #~ from pprint import pprint
    #~ pprint(d)