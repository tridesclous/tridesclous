from tridesclous import *
from tridesclous.tools import download_probe
import  pyqtgraph as pg




def test_ProbeGeometryView():
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
