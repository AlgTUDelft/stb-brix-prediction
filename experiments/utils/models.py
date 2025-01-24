
import tensorflow as tf
from tensorflow import keras
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Lambda
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import InputLayer, Reshape, RepeatVector
from keras import optimizers, initializers
from keras.callbacks import History, EarlyStopping, ModelCheckpoint

def create_models(input_dim):

    model0 = Sequential()
    model0.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                activity_regularizer=tf.keras.regularizers.l2(0.1), input_dim=input_dim))
    model0.add(Dropout(0.2))
    model0.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                activity_regularizer=tf.keras.regularizers.l2(0.1)))
    model0.add(Dropout(0.1))
    model0.add(Dense(1, activation='linear'))
    model0.save('models/empty_fc1.h5')


    model1 = Sequential()
    model1.add(Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                activity_regularizer=tf.keras.regularizers.l2(0.1), input_dim=input_dim))
    model1.add(Dropout(0.2))
    model1.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                activity_regularizer=tf.keras.regularizers.l2(0.1)))
    model1.add(Dropout(0.1))
    model1.add(Dense(1, activation='linear'))
    model1.save('models/empty_fc2.h5')


    model2 = Sequential()
    model2.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                activity_regularizer=tf.keras.regularizers.l2(0.1), input_dim=input_dim))
    model2.add(Dropout(0.2))
    model2.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                activity_regularizer=tf.keras.regularizers.l2(0.1)))
    model2.add(Dropout(0.1))
    model2.add(Dense(1, activation='linear'))
    model2.save('models/empty_fc3.h5')   


def create_expd_models(input_dim, attr_key, encoder, r1=0.01,r2=0.1,d1=0.2,d2=0.1,series=0):
    modelb = Sequential()
    modelb.add(Dense(256, activation='relu', input_dim=input_dim, \
                                             kernel_regularizer=tf.keras.regularizers.l1(r1),\
                                             activity_regularizer=tf.keras.regularizers.l2(r2)))
    modelb.add(Dense(256, activation='relu'))
    modelb.add(Dropout(d1))
    modelb.add(Dense(32, activation='relu'))
    modelb.add(Dropout(0.1))
    modelb.add(Dense(1, activation='linear'))
    modelb.save(f'models/{attr_key}_r_empty_fcb_{encoder}_{series}.h5')

    model0 = Sequential()
    model0.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(r1),
                                activity_regularizer=tf.keras.regularizers.l2(r2), input_dim=input_dim))
    model0.add(Dropout(d1))
    model0.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(r1),
                                activity_regularizer=tf.keras.regularizers.l2(r2)))
    model0.add(Dropout(d2))
    model0.add(Dense(1, activation='linear'))
    model0.save(f'models/{attr_key}_r_empty_fc1_{encoder}_{series}.h5')

    model1 = Sequential()
    model1.add(Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(r1),
                                activity_regularizer=tf.keras.regularizers.l2(r2), input_dim=input_dim))
    model1.add(Dropout(d1))
    model1.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(r1),
                                activity_regularizer=tf.keras.regularizers.l2(r2)))
    model1.add(Dropout(d2))
    model1.add(Dense(1, activation='linear'))
    model1.save(f'models/{attr_key}_r_empty_fc2_{encoder}_{series}.h5')

    model2 = Sequential()
    model2.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(r1),
                                activity_regularizer=tf.keras.regularizers.l2(r2), input_dim=input_dim))
    model2.add(Dropout(d1))
    model2.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(r1),
                                activity_regularizer=tf.keras.regularizers.l2(r2)))
    model2.add(Dropout(d2))
    model2.add(Dense(1, activation='linear'))
    model2.save(f'models/{attr_key}_r_empty_fc3_{encoder}_{series}.h5')   