from pathlib import Path
import os
from glob import glob
from pickletools import int4
from timeit import repeat

import pandas as pd
import numpy as np
import copy

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Lambda
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import InputLayer, Reshape, RepeatVector
from keras import optimizers, initializers
from keras.callbacks import History, EarlyStopping, ModelCheckpoint

import random
import datetime

from utils.utils import *
from utils.models import *

basePath = Path(os.getcwd()).parent
dataPath = os.path.join(basePath, 'example_data')

# Parameters
attr = 'brix'
key = f'{attr}-expc-demo'

epochs = 200
seed = 2022
batch_size = 16
model = '0'

num_seeds = 5
random.seed(seed)
seedlist = random.sample(range(4000), num_seeds)

# Load data
df = pd.read_csv(os.path.join(dataPath, 'Strawberry_Measurements_with_Seg_Connections_mtd1.csv'))

df = df[df.RGB_Camera>0].reset_index(drop=True)
df['ID']=df['Strawberry_ID'].str.replace('.',',')
df[['dd','mm','br','nr']]=df['ID'].str.split(",",expand=True,)
df[['dd','mm']] = df[['dd','mm']].astype(int)
df[['yy']] = (np.ones(shape=(df.shape[0],1))*2021).astype(int)
df['date'] = df.apply(lambda x: datetime.date(x['yy'], x['mm'], x['dd']), axis=1)
df['formatted_date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df.formatted_date.apply(lambda x: x.dayofyear)
df['week_of_year'] = df.formatted_date.apply(lambda x: x.weekofyear)
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Sugariness'])

env_data = pd.read_csv(os.path.join(dataPath,'env_dataing.csv'), index_col=0)
if 'week_of_year' in env_data.columns:
    env_data = env_data.drop(columns='week_of_year')
cols = env_data.columns.tolist()
cols = ['day_of_year'] + [col for col in env_data.columns if col!='day_of_year' and not col.__contains__('PH')]
env_w = copy.copy(env_data[cols])

# merge inputs
images = []
labels = []
env = []

for idx in df.index:
    img_file = os.path.join(dataPath, 'Segments', df['segment_file'][idx])
    d = df.iloc[idx]['day_of_year']
    if os.path.exists(img_file):
        im = Image.open(img_file)
        im_array = np.array(im)
        im_array[im_array[:,:,3]==0]=0
        images.append(im_array)
        labels.append(df['Sugariness'][idx])
        env.append(env_w[env_w.day_of_year==d].values.reshape(-1)[1:])
    else:
        print(df['segment_file'][idx], 'missing')

labels = np.stack(labels)
images = np.stack(images)
env = np.stack(env)

#%%
n_env = (env_w.shape[1]-1)/5

info_2_col = dict()
info_2_col['l1w'] = np.arange(n_env*2+1,n_env*3+1) -1
info_2_col['w0'] = np.arange(n_env*3+1,n_env*4+1) -1
info_2_col['f1w'] = np.arange(n_env*4+1,n_env*5+1) -1

info_2_col_env_combi = {}

w0_f1 = np.append(info_2_col['w0'],info_2_col['f1w'])
info_2_col_env_combi['l1-f1'] = np.append(info_2_col['l1w'],w0_f1)


#%%
import matplotlib.pyplot as plt
import time

from numpy.random import seed
seed(1)

import tensorflow
tensorflow.random.set_seed(2022)

from sklearn.decomposition import PCA

all_errors = {}

encoder = keras.models.load_model(os.path.join(dataPath, 'example_encoder.h5'))
im_train, _ = get_data_ae(np.arange(labels.shape[0]), images, labels, encoder=encoder)

errors=[]

for info in info_2_col_env_combi.keys():
    envs = env[:,info_2_col_env_combi[info].astype(int)]

    for s in seedlist:
        im_train, y_train, im_test, y_test, im_val, y_val, yScaler, i_val, \
        env_train, env_test, env_val, envScaler \
        = generate_dataset_c(images, labels, envs, val_per=0.2, test_per=0.15, seed=s, encoder=encoder)

        pc = PCA(n_components = im_train.shape[1]).fit(im_train.reshape(im_train.shape[0],-1))
        im_train = pc.transform(im_train.reshape(im_train.shape[0],-1))
        im_test = pc.transform(im_test.reshape(im_test.shape[0],-1))
        im_val = pc.transform(im_val.reshape(im_val.shape[0],-1))



        y_val_beforeScale = yScaler.inverse_transform(y_val.reshape(-1,1))
        bins = np.linspace(-5,5,11)

        timestr = time.strftime("%Y%m%d-%H%M")  
        X_train = np.hstack((im_train.reshape(im_train.shape[0],-1),env_train))
        X_test = np.hstack((im_test.reshape(im_test.shape[0],-1),env_test))
        X_val = np.hstack((im_val.reshape(im_val.shape[0],-1),env_val))
        title = 'predicted-with-{}enc_pca+env-{}-c'.format(encoder,info)

        model = keras.models.load_model('empty_fc{}_1c.h5'.format(m))
        mc = ModelCheckpoint('brix_with_{}enc+env-{}_1c.h5'.format(encoder, info), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss='mse')

        history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=16, validation_data=(X_test, y_test), callbacks=[es, mc])

        y_pred = yScaler.inverse_transform(model.predict(X_val))
        err = y_val_beforeScale.reshape(-1)-y_pred.reshape(-1)
        errors.append(err)
                
err_df = pd.DataFrame(errors)
err_df.transpose().to_csv(f'results/pred_errors_{key}.csv')