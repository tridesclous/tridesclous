from pyacq.devices import OpenEphysGUIRelay

from tridesclous.online.launcher import start_online_pyacq_buffer_demo,  start_online_openephys

import pyqtgraph as pg





def test_start_online_pyacq_buffer_demo():
    start_online_pyacq_buffer_demo()



def test_start_online_openephys():
    start_online_openephys(prb_filename = 'probe_openephys_16ch.prb', workdir='/home/samuel/Desktop/test_tdconlinewindow_openephys/')
    #~ start_online_openephys(prb_filename = 'probe_openephys_16ch.prb')
    #~ start_online_openephys()
    
    


if __name__ == '__main__':
    #~ test_start_online_pyacq_buffer_demo()
    test_start_online_openephys()
