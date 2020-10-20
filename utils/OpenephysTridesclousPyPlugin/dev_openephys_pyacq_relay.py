import zmq
import json
from pprint import pprint
import pyacq
print(pyacq.__file__)

#~ url = 'tcp://127.0.0.1:*'
url = 'tcp://127.0.0.1:20000'

context = zmq.Context.instance()
socket = context.socket(zmq.REQ)
socket.connect(url)

# get config
msg = b'config'
socket.send(msg)
msg = socket.recv()
print(msg)
stream_params = json.loads(msg.decode())
pprint(stream_params)

# input stream
stream = pyacq.InputStream()
stream.connect(stream_params)
stream.set_buffer(size=stream_params['buffer_size'])

print('ici')

# start
msg = b'start'
socket.send(msg)
msg = socket.recv()
assert msg == b'ok'
print('yep')

# read loop
for i in range(50):
    pos, data = stream.recv(return_data=True)
    print(pos)
    #~ data = stream.get_data(pos-100, pos)
    print(data.shape, data.dtype)

msg = b'stop'
socket.send(msg)
msg = socket.recv()
assert msg == b'ok'


socket.close()