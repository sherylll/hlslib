"""
# with a Sequential model, get intermediate results
get_layer_output = K.function([model.layers[0].input],[model.layers[0].output])
output = get_layer_output([x])[0]
"""

import numpy as np
from keras.models import model_from_json
from keras.datasets import mnist
from keras import backend as K
import keras

with open('conv2d.json') as f:
    json_str = f.read()
model = model_from_json(json_str)
model.load_weights('conv2d.h5')
weights = model.get_weights()

np.savetxt('w_conv.h',np.transpose(weights[0], (3,2,0,1)).reshape((4,16)), delimiter=',', newline=',\n', header = 'float w_conv[4][1][16]={',footer='};', comments='')
np.savetxt('b_conv.h', weights[1], newline=',', header = 'float b_conv[4]={',footer='};', comments='')
np.savetxt('w_fc0.h',np.transpose(weights[2]), delimiter=',', newline=',\n', header = 'float w_fc0[32][676]={',footer='};', comments='')
np.savetxt('b_fc0.h',weights[3],newline=',', header = 'float b_fc0[32]={',footer='};', comments='')
np.savetxt('w_fc1.h',np.transpose(weights[4]), delimiter=',', newline=',\n', header = 'float w_fc1[10][32]={',footer='};', comments='')
np.savetxt('b_fc1.h',weights[5],newline=',', header = 'float b_fc1[10]={',footer='};', comments='')

