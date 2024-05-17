from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

def RML_Model(input_dim, nb_classes):
    inputs = Input([input_dim, 1])
    layer0 = Flatten()(inputs)
    layer1 = Dense(128, activation='relu', name="layer1")(layer0)
    # layer1 = Dropout(0.3)(layer1)

    layer2 = Dense(128, activation='relu', name="layer2")(layer1)
    # layer2 = Dropout(0.3)(layer2)

    layer3 = Dense(128, activation='relu', name="layer3")(layer2)
    # layer3 = Dropout(0.3)(layer3)

    layer4 = Dense(128, activation='relu', name="layer4")(layer3)
    # layer4 = Dropout(0.3)(layer4)

    layer5 = Dense(128, activation='relu', name="layer5")(layer4)
    # layer5 = Dropout(0.3)(layer5)

    layer6 = Dense(128, activation='relu', name="layer6")(layer5)
    # layer6 = Dropout(0.3)(layer6)

    layer7 = Dense(nb_classes, activation='softmax', name="layer7")(layer6)

    model = Model(input=inputs, output=layer7)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model = models.Sequential()
    model.add(Reshape([1] + in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(256, (1, 3), padding='valid', input_shape=(1, 2, 128), activation="relu",
                     kernel_initializer='glorot_uniform'))
    # model.add(Conv2D(512, (1, 3), padding='valid', input_shape=(1, 2, 128), activation="relu",
    #                  kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(80, (1, 3), strides=1, padding="valid", input_shape=(1, 2, 128), activation="relu",
                     kernel_initializer='glorot_uniform'))
    # model.add(DepthwiseConv2D(1, (1, 3), padding="valid", input_shape=(1, 2, 128), activation="relu",
    #                  kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(dr))
    model.add(Dense(len(classes), kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.build()
    model.summary()