"""
creates simple cnn without padding
input should be channel-first
only work for single-channel input (such as MNIST)
"""
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32').reshape(60000,1,28,28)/255
x_test = x_test.astype('float32').reshape(10000,1,28,28)/255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

n_filters = 4

# inputs = Input(shape=(1, 28, 28))
# layer = Conv2D(n_filters, (4, 4), strides=(2,2), activation='relu', input_shape=x_train.shape[1:], name="conv2d", data_format='channels_first')(inputs)
# layer = Flatten(name='flatten')(layer)
# layer = Dense(32, activation='relu', name='fc0')(layer)
# layer = Dropout(0.5)(layer)
# layer = Dense(10, activation='softmax')(layer)
# model = Model(inputs=inputs, outputs=layer)

model = Sequential()
model.add(Conv2D(n_filters, (4, 4), strides=(2,2), activation='relu', input_shape=x_train.shape[1:], name="conv2d", data_format='channels_first'))
model.add(Flatten(name='flatten'))
model.add(Dense(32, activation='relu', name='fc0'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)

weights = model.get_weights()
for w in weights:
    print(w.shape)
    
json_str = model.to_json()
with open('conv2d_weights/conv2d.json', 'w+') as f:
    f.write(json_str)
model.save_weights('conv2d_weights/conv2d.h5')


# (w, w, ichan, ochan) -> (ochan, ichan, w, w)
np.savetxt('conv2d_weights/w_conv.h',np.transpose(weights[0], (3,2,0,1)).reshape((4,16)), delimiter=',', newline=',\n', header = 'float w_conv[4][1][16]={',footer='};', comments='')
np.savetxt('conv2d_weights/b_conv.h', weights[1], newline=',', header = 'float b_conv[4]={',footer='};', comments='')
np.savetxt('conv2d_weights/w_fc0.h',np.transpose(weights[2]), delimiter=',', newline=',\n', header = 'float w_fc0[32][676]={',footer='};', comments='')
np.savetxt('conv2d_weights/b_fc0.h',weights[3],newline=',', header = 'float b_fc0[32]={',footer='};', comments='')
np.savetxt('conv2d_weights/w_fc1.h',np.transpose(weights[4]), delimiter=',', newline=',\n', header = 'float w_fc1[10][32]={',footer='};', comments='')
np.savetxt('conv2d_weights/b_fc1.h',weights[5],newline=',', header = 'float b_fc1[10]={',footer='};', comments='')

