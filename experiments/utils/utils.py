import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Lambda

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from numpy.random import seed

seed(1)
tf.random.set_seed(2022)

def transform_img(images, crop_x0=50, crop_x1=250, crop_y0=50, crop_y1=250, crop_c0=0, crop_c1=3, enc=None):
    if images.max()>1:
        images = images/255
    if enc is not None:
        encoder = keras.models.load_model(basePath/'extract-image-feature'/'Image_Encoding'/'Encoders'/f'enc200-{enc}s.h5', compile=False)
        if len(images.shape)==3:
            X = encoder.predict(np.array([images[crop_x0:crop_x1,crop_y0:crop_y1,crop_c0:crop_c1]])).reshape(1,-1)
        else:
            X = encoder.predict(images[:,crop_x0:crop_x1,crop_y0:crop_y1,crop_c0:crop_c1]).reshape(images.shape[0],-1)
        return X
    else:
        return images[:,crop_x0:crop_x1,crop_y0:crop_y1,crop_c0:crop_c1].reshape(images.shape[0],-1)

def get_splited_data(iids, images, labels, supplement_data, encoder=None, sel_pieces=1):
    im_id = np.repeat(iids, sel_pieces) + np.tile(np.arange(sel_pieces), iids.shape[0])
    X = transform_img(images[im_id], enc=encoder)
    y = np.repeat(labels[iids], sel_pieces)
    sups = np.repeat(supplement_data[iids], sel_pieces, axis=0)
    return X,y,sups


def get_data_ae(iids, images, labels, encoder=None, sel_pieces=1):
    im_id = np.repeat(iids, sel_pieces) + np.tile(np.arange(sel_pieces), iids.shape[0])
    X = images[im_id]/255
    if encoder is not None:
        X = encoder.predict(X[:,50:250,50:250,:3])#.reshape(X.shape[0],-1)
    else:
        X = X[:,50:250,50:250,:3]
    y = np.repeat(labels[iids], sel_pieces)
    return X,y

def get_data_ae(iids, images, labels, encoder=None, sel_pieces=1):
    im_id = np.repeat(iids, sel_pieces) + np.tile(np.arange(sel_pieces), iids.shape[0])
    X = images[im_id]/255
    if encoder is not None:
        X = encoder.predict(X[:,50:250,50:250,:3])#.reshape(X.shape[0],-1)
    else:
        X = X[:,50:250,50:250,:3]
    y = np.repeat(labels[iids], sel_pieces)
    return X,y


def generate_dataset(images, labels, val_per=0.2, test_per=0.15, seed=3407, sel_pieces=1, encoder=None):
    iids = np.arange(labels.shape[0])
    i_train0, i_val = train_test_split(iids, test_size=val_per, random_state=seed, shuffle=True)
    i_train, i_test = train_test_split(i_train0, test_size=test_per, random_state=seed*2, shuffle=True)

    X_train, y_train = get_data_ae(i_train0, images, labels, encoder, sel_pieces)
    X_test, y_test = get_data_ae(i_test, images, labels, encoder, sel_pieces)
    X_val, y_val = get_data_ae(i_val, images, labels, encoder, sel_pieces)
    
    yScaler = StandardScaler()
    yScaler = yScaler.fit(y_train.reshape(-1,1))
    y_train = yScaler.transform(y_train.reshape(-1,1))
    y_test = yScaler.transform(y_test.reshape(-1,1))
    y_val = yScaler.transform(y_val.reshape(-1,1))

    return X_train, y_train, X_val, y_val, X_test, y_test, yScaler



def generate_dataset_c(images, labels, env, val_per=0.2, test_per=0.15, seed=3407, sel_pieces=1, encoder=None):
    iids = np.arange(labels.shape[0])
    i_train0, i_val = train_test_split(iids, test_size=val_per, random_state=seed, shuffle=True)
    i_train, i_test = train_test_split(i_train0, test_size=test_per, random_state=seed*2, shuffle=True)

    X_train, y_train = get_data_ae(i_train, images, labels, encoder, sel_pieces)
    X_test, y_test = get_data_ae(i_test, images, labels, encoder, sel_pieces)
    X_val, y_val = get_data_ae(i_val, images, labels, encoder, sel_pieces)
    
    yScaler = StandardScaler()
    yScaler = yScaler.fit(y_train.reshape(-1,1))
    y_train = yScaler.transform(y_train.reshape(-1,1))
    y_test = yScaler.transform(y_test.reshape(-1,1))
    y_val = yScaler.transform(y_val.reshape(-1,1))

    env_train = np.repeat(env[i_train], sel_pieces, axis=0)
    env_test = np.repeat(env[i_test], sel_pieces, axis=0)
    env_val = np.repeat(env[i_val], sel_pieces, axis=0)

    envScaler = StandardScaler()
    envScaler = envScaler.fit(env_train)
    env_train = envScaler.transform(env_train)
    env_test = envScaler.transform(env_test)
    env_val = envScaler.transform(env_val)

    return X_train, y_train, X_test, y_test, X_val, y_val, yScaler, i_val, env_train, env_test, env_val, envScaler



def generate_dataset_d(images, labels, supplement_data, val_per=0.2, test_per=0.15, seed=3407, sel_pieces=3, encoder=None):
    iids = np.arange(labels.shape[0])
    i_train0, i_val = train_test_split(iids, test_size=val_per, random_state=seed, shuffle=True)
    i_train, i_test = train_test_split(i_train0, test_size=test_per, random_state=seed*2, shuffle=True)

    X_train, y_train, sup_train = get_splited_data(i_train, images, labels, supplement_data, encoder, sel_pieces)
    X_test, y_test, sup_test = get_splited_data(i_test, images, labels, supplement_data, encoder, sel_pieces)
    X_val, y_val, sup_val = get_splited_data(i_val, images, labels, supplement_data, encoder, sel_pieces)
    
    yScaler = StandardScaler()
    yScaler = yScaler.fit(y_train.reshape(-1,1))
    y_train = yScaler.transform(y_train.reshape(-1,1))
    y_test = yScaler.transform(y_test.reshape(-1,1))
    y_val = yScaler.transform(y_val.reshape(-1,1))

    return X_train, y_train, X_test, y_test, X_val, y_val, yScaler, sup_train, sup_test, sup_val, i_val 
