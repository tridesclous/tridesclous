"""



"""
try:
    import pyacq
    #test pyacq version
    import packaging.version
    assert packaging.version.Version(pyacq.__version__)>='0.2.0-dev'
    HAVE_PYACQ = True
except:
    HAVE_PYACQ = False


if HAVE_PYACQ:
    from .onlinepeeler import OnlinePeeler
    from .onlinetools import make_empty_catalogue, lighter_catalogue
    from .onlinetraceviewer import OnlineTraceViewer
    from .onlinewindow import TdcOnlineWindow
    from .onlinewaveformhistviewer import OnlineWaveformHistViewer
    from .launcher import make_pyacq_device_from_buffer, start_online_window, start_online_pyacq_buffer_demo, start_online_openephys
    from .probeactivityviewer import ProbeActivityViewer

