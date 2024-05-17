from keras.layers import *#BatchNormalization, Conv2D, ZeroPadding2D, Dense,Dropout, Flatten,Reshape,
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
# from keras.layers import Dense, Input, Dropout
from keras.optimizers import *
def build_model_actual(batch_normalization,activation,in_shp,dr,classes):
    model = Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, 1, 3, padding='valid', activation=activation, name="conv1", kernel_initializer='glorot_uniform'))
    if batch_normalization: model.add(BatchNormalization())
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, 1, 3, padding="valid", activation=activation, name="conv2", kernel_initializer='glorot_uniform'))
    if batch_normalization: model.add(BatchNormalization())
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    return model

def build_model_actual_3(batch_normalization, activation, in_shp, dr, classes):
    model = Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(256, (1, 3), padding='valid', input_shape=(1, 2, 128), activation=activation, kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(80, (1, 3), strides=1, padding="valid", input_shape=(1, 2, 128), activation=activation, kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(dr))
    model.add(Dense( len(classes), kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    return model

def build_model_modified_ConvNet(batch_normalization,activation,in_shp,dr,classes):
    model = Sequential()
    model.add(Reshape([1] + in_shp, input_shape=in_shp))

    model.add(Conv2D(512, (1, 2), padding='same', input_shape=(1, 2, 128), activation=activation,
                     kernel_initializer='glorot_uniform'))

    if batch_normalization: model.add(BatchNormalization())
    model.add(Dropout(dr))
    model.add(Conv2D(512, (1, 2), padding='same', input_shape=(1, 2, 512), activation=activation,
                     kernel_initializer='glorot_uniform'))

    if batch_normalization: model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dr))

    model.add(Dense(len(classes), activation= 'softmax'))
    model.add(Reshape([len(classes)]))

    return model

def build_model_modified_MobileNet(batch_normalization,activation,in_shp,dr,classes):
    model = Sequential()
    model.add(Reshape([1] + in_shp, input_shape=in_shp))

    model.add(Conv2D(256, (1, 2), padding='same', input_shape=(1, 2, 128), activation=activation,
                     kernel_initializer='glorot_uniform'))

    model.add(DepthwiseConv2D(kernel_size=(1, 2), padding='same', input_shape=(1, 2, 256), activation=None,
                     depth_multiplier=2, kernel_initializer='glorot_uniform'))

    if batch_normalization: model.add(BatchNormalization())
    model.add(Dropout(dr))

    model.add(Conv2D(256, (1, 2), padding='same', input_shape=(1, 2, 512), activation=activation,
                     kernel_initializer='glorot_uniform'))

    model.add(DepthwiseConv2D(kernel_size=(1, 2), padding='same', input_shape=(1, 2, 256), activation=None,
                              depth_multiplier=2, kernel_initializer='glorot_uniform'))
    if batch_normalization: model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(80, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dr))

    model.add(Dense(len(classes), activation= 'softmax'))
    model.add(Reshape([len(classes)]))

    return model

def build_model_modified_Separable(batch_normalization,activation,in_shp,dr,classes):
    model = Sequential()
    model.add(Reshape([1] + in_shp, input_shape=in_shp))

    model.add(Conv2D(256, (1, 2), padding='same', input_shape=(1, 2, 128), activation=activation,
                     kernel_initializer='glorot_uniform'))

    model.add(SeparableConv2D(256, kernel_size=(1, 2), padding='same', input_shape=(1, 2, 256), activation=None,
                     depth_multiplier=2, kernel_initializer='glorot_uniform'))

    if batch_normalization: model.add(BatchNormalization())
    model.add(Dropout(dr))

    model.add(Conv2D(256, (1, 2), padding='same', input_shape=(1, 2, 512), activation=activation,
                     kernel_initializer='glorot_uniform'))

    model.add(SeparableConv2D(80, kernel_size=(1, 2), padding='same', input_shape=(1, 2, 256), activation=None,
                              depth_multiplier=2, kernel_initializer='glorot_uniform'))
    if batch_normalization: model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(80, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dr))

    model.add(Dense(len(classes), activation= 'softmax'))
    model.add(Reshape([len(classes)]))

    return model

# def build_model(batch_normalization,activation,in_shp,dr,classes):
#     model = Sequential()
#     model.add(Reshape([1] + in_shp, input_shape=in_shp))
#
#     model.add(ZeroPadding2D((0, 2)))
#     model.add(Conv2D(2048, (1, 3), padding='valid', input_shape=(1, 2, 128), activation=activation,
#                      kernel_initializer='glorot_uniform'))
#     if batch_normalization: model.add(BatchNormalization())
#     model.add(Dropout(dr))
#     model.add(ZeroPadding2D((0, 2)))
#     model.add(Conv2D(512, (1, 3), padding='valid', input_shape=(1, 2, 128), activation=activation,
#                      kernel_initializer='glorot_uniform'))
#     if batch_normalization: model.add(BatchNormalization())
#     # model.add(MaxPooling2D(2, 2))
#     model.add(Dropout(dr))
#
#
#     model.add(Flatten())
#     model.add(Dense(256, activation=activation, kernel_initializer='he_normal'))
#     model.add(Dropout(dr))
#
#     model.add(Dense(len(classes), activation= 'softmax'))#,kernel_initializer='he_normal'))
#     # model.add(Activation('softmax'))
#     model.add(Reshape([len(classes)]))
#
#     return model

def build_model_modified_Mixnet(batch_normalization,activation,in_shp,dr,classes):
    model = Sequential()
    model.add(Reshape([1] + in_shp, input_shape=in_shp))

    # model.add(Conv2D(256, (1, 2), padding='same', input_shape=(1, 2, 128), activation=activation,
    #                  kernel_initializer='glorot_uniform'))

    model.add(SeparableConv2D(512, kernel_size=(1, 3), padding='same', input_shape=(1, 2, 256), activation=None,
                     depth_multiplier=2, kernel_initializer='glorot_uniform'))

    if batch_normalization: model.add(BatchNormalization())
    model.add(Dropout(dr))

    model.add(Conv2D(256, (1, 2), padding='same', input_shape=(1, 2, 512), activation=activation,
                     kernel_initializer='glorot_uniform'))

    model.add(SeparableConv2D(256, kernel_size=(1, 3), padding='same', input_shape=(1, 2, 256), activation=None,
                              depth_multiplier=2, kernel_initializer='glorot_uniform'))
    model.add(Conv2D(128, (1, 2), padding='same', input_shape=(1, 2, 128), activation=activation,
                     kernel_initializer='glorot_uniform'))
    model.add(SeparableConv2D(128, kernel_size=(1, 5), padding='same', input_shape=(1, 2, 256), activation=None,
                              depth_multiplier=2, kernel_initializer='glorot_uniform'))
    model.add(SeparableConv2D(80, kernel_size=(1, 7), padding='same', input_shape=(1, 2, 256), activation=None,
                              depth_multiplier=2, kernel_initializer='glorot_uniform'))

    if batch_normalization: model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(80, activation=activation, kernel_initializer='he_normal'))
    model.add(Dropout(dr))

    model.add(Dense(len(classes), activation= 'softmax'))
    model.add(Reshape([len(classes)]))

    return model