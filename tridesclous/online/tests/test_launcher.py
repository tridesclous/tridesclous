import pytest

from tridesclous.online import HAVE_PYACQ
from tridesclous.tests.testingtools import ON_CI_CLOUD

if HAVE_PYACQ:
    from tridesclous.online import *
    import pyacq
    from pyacq.devices import OpenEphysGUIRelay

    from tridesclous.online.launcher import start_online_pyacq_buffer_demo,  start_online_openephys

import pyqtgraph as pg



@pytest.mark.skipif(not HAVE_PYACQ, reason='no pyacq')
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_start_online_pyacq_buffer_demo():
    start_online_pyacq_buffer_demo()


@pytest.mark.skipif(not HAVE_PYACQ, reason='no pyacq')
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_start_online_openephys():
    # with 16ch
    #~ prb_filename = 'probe_openephys_16ch.prb'
    #~ workdir = '/home/samuel/Desktop/test_tdconlinewindow_openephys_oe16ch/'
    #~ workdir = None

    # with meraec 32ch
    prb_filename = '/home/samuel/Documents/DataSpikeSorting/mearec/openephys_mearec_32ch/openephys_mearec_32ch.prb'
    workdir = '/home/samuel/Desktop/test_tdconlinewindow_openephys_mearec32ch/'


    start_online_openephys(prb_filename=prb_filename, workdir=workdir)



if __name__ == '__main__':
    test_start_online_pyacq_buffer_demo()
    #~ test_start_online_openephys()
    #~ pass

