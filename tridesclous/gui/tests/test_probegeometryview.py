from tridesclous.tools import download_probe


import pytest
from tridesclous.tests.testingtools import ON_CI_CLOUD
from tridesclous.gui.tests.testingguitools import HAVE_QT5

    

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ProbeGeometryView():
    from tridesclous.gui import mkQApp, ProbeGeometryView
    
    local_dirname = ''
    
    probe_filename = download_probe(local_dirname, 'mea_256', origin='spyking-circus')
    d = {}
    exec(open(probe_filename).read(), None, d)
    channel_groups = d['channel_groups']
    
    print(channel_groups)
    
    app = mkQApp()
    view = ProbeGeometryView(channel_groups=channel_groups)
    view.show()
    
    if __name__ == '__main__':
        app.exec_()
    
    


if __name__ == '__main__':
    test_ProbeGeometryView()
