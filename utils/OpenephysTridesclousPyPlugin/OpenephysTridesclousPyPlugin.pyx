import sys
import os

import numpy as np
cimport numpy as np
from cython cimport view
import zmq
import json
import time
import threading
import pyacq
from pprint import pprint


import pyacq



isDebug = False

DEFAULT_PORT = 20000



class ThreadConfig(threading.Thread):
    def __init__(self, parent=None, port=DEFAULT_PORT, stream_params={},**kargs):
        threading.Thread.__init__(self, **kargs)
        
        self.parent = parent
        self.port = port
        self.stream_params = stream_params
        print('*'*20)
        pprint(self.stream_params)
        print('*'*20)
        
        url = 'tcp://127.0.0.1:{}'.format(self.port)
        
        context = zmq.Context.instance()
        self.socket_info = context.socket(zmq.REP)
        self.socket_info.linger = 1000  # don't let socket deadlock when exiting
        self.socket_info.bind(url)

        
    def run(self):
        #~ print('ThreadConfig.run')
        self.is_listening = True
        while self.is_listening:
            if self.socket_info.poll(timeout=200) > 0:   # 200ms
                request = self.socket_info.recv().decode()
                print('request', request)
                
                #~ print('recv', msg)
                if request == 'config':
                    msg = json.dumps(self.stream_params)
                elif request == 'start':
                    msg = 'ok'
                    self.parent.start_stream()
                elif request == 'stop':
                    msg = 'ok'
                    self.parent.stop_stream()
                else:
                    msg = 'what??'
                self.socket_info.send(msg.encode())
                
            else:
                #~ print('sleep ThreadConfig')
                time.sleep(0.5)
    
    def stop(self):
        self.is_listening = False


class OpenephysTridesclousPyPlugin(object):
    def __init__(self, port=DEFAULT_PORT):
        """initialize object data"""
        self.port = port
        
        self.Enabled = 1
        self.samplingRate = 0.
        self.chan_enabled = []
        
        print('*'*20)
        print('Plugin pyacq tridesclous')
        print('*'*20)
    
    def __del__(self):
        print('__del__')
        if hasattr(self, 'thread_config'):
            self.thread_config.stop()
            self.thread_config.wait()

    def startup(self, nchans, srate, states):
        """to be run upon startup"""
        print('*'*20)
        print('startup pyacq oe')
        print('*'*20)
        
        
        self.mutex = threading.Lock()
        self.is_running= False
        
        # create stream 'sharedmem' with 4s of buffer
        p = dict(
            protocol='tcp',
            interface='127.0.0.1',
            port='*',
            transfermode='sharedmem',
            streamtype='analogsignal',
            dtype='float32',
            shape=(-1, nchans),
            axisorder=None,
            buffer_size=int(4 * srate),
            compression='',
            scale=None,
            offset=None,
            units='',
            sample_rate=float(srate),
            double=False,
            fill=0.,
        )
        self.output = pyacq.OutputStream()
        self.output.configure(**p)
        #~ pprint(self.output.params)
        
        
        self.thread_config = ThreadConfig(parent=self, port=self.port, stream_params=dict(self.output.params))
        self.thread_config.start()
        
        self.update_settings(nchans, srate)
        for chan in range(nchans):
            if not states[chan]:
                self.channel_changed(chan, False)
    
    def plugin_name(self):
        """tells OE the name of the program"""
        return "OpenephysTridesclousPyPlugin"

    def is_ready(self):
        """tells OE everything ran smoothly"""
        return self.Enabled

    def param_config(self):
        """return button, sliders, etc to be present in the editor OE side"""
        configs = [
                #~ ("toggle", "Enabled", True)
                #~ ("int_set", "port", DEFAULT_PORT)
                ]
        return configs

    def update_settings(self, nchans, srate):
        """handle changing number of channels and sample rates"""
        print('*'*20)
        print('update_settings')
        print('*'*20)
        self.samplingRate = srate

        old_nchans = len(self.chan_enabled)
        if old_nchans > nchans:
            del self.chan_enabled[nchans:]
        elif len(self.chan_enabled) < nchans:
            self.chan_enabled.extend([True] * (nchans - old_nchans))

    def channel_changed(self, chan, state):
        """do something when channels are turned on or off in PARAMS tab"""
        self.chan_enabled[chan] = state

    def bufferfunction(self, n_arr):
        """Access to voltage data buffer. Returns events""" 
        with self.mutex:
            #~ print('self.is_running', self.is_running)
            if self.is_running:
                self.output.send(n_arr.T)
        
        events = []
        return events

    def handleEvents(self, eventType,sourceID,subProcessorIdx,timestamp,sourceIndex):
        """handle events passed from OE"""
        print('handleEvents')

    def handleSpike(self, electrode, sortedID, n_arr):
        """handle spikes passed from OE"""
        print('handleSpike')
    
    def start_stream(self):
        print('start_stream')
        
        with self.mutex:
            self.is_running= True
    
    def stop_stream(self):
        print('stop_stream')
        with self.mutex:
            self.is_running= False

pluginOp = OpenephysTridesclousPyPlugin()

include '../plugin.pyx'
