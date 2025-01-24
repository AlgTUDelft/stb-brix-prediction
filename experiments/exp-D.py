from pathlib import Path
import os
from glob import glob
from timeit import repeat
from pickletools import int4

import pandas as pd
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
import time

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Lambda
from keras.callbacks import History, EarlyStopping, ModelCheckpoint

import datetime
import random

from utils.utils import *
from utils.models import *

basePath = Path(os.getcwd()).parent
dataPath = os.path.join(basePath, 'example_data')

# Parameters
attr = 'brix'
key = f'{attr}-expd-demo'

epoch = 200
seed = 2022
batch_size = 16
model = '0'

num_seeds = 5
random.seed(seed)
seed_fraction = 0
seedlist = random.sample(range(4000), num_seeds)

env_agg = 'rolling'
sel_model = '_KRR-a100-d1_'

#%%
df = pd.read_csv(os.path.join(dataPath, 'Strawberry_Measurements_with_Seg_Connections_mtd1.csv'))
df['ID']=df['ID'].str.replace('.',',')
df[['dd','mm','br','nr']]=df['ID'].str.split(",",expand=True,)
df[['dd','mm']] = df[['dd','mm']].astype(int)
df[['yy']] = (np.ones(shape=(df.shape[0],1))*2021).astype(int)
df['date'] = df.apply(lambda x: datetime.date(x['yy'], x['mm'], x['dd']), axis=1)
df['formatted_date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df.formatted_date.apply(lambda x: x.dayofyear)
df['week_of_year'] = df.formatted_date.apply(lambda x: x.weekofyear)
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Sugariness'])


# df_pred = pd.read_csv(Path(os.getcwd())/'old-data'/f'Prediction_Results_{env_agg}.csv', index_col=0)
df_pred = pd.read_csv(os.path.join(dataPath, 'example_prediction_res.csv.csv'), index_col=0)
sel_col = list(df_pred.columns[:12]) + [col for col in df_pred.columns[12:] if col.__contains__(sel_model)]
df_pred = df_pred[sel_col]

#%%
encoder = 'm2m2'
all_mlp = [f'model{m}' for m in ['b',1,2,3]]
all_mlp22 = [f'model{m}' for m in [2,3]]

# merge inputs
images = []
labels = []
sup = []

for idx in df.index:
    img_file = os.path.join(dataPath, 'Segments', df['segment_file'][idx])
    d = df.iloc[idx]['day_of_year']
    if os.path.exists(img_file):
        im = Image.open(img_file)
        im_array = np.array(im)
        im_array[im_array[:,:,3]==0]=0
        images.append(im_array)
        labels.append(df['Sugariness'][idx])
        sup.append(df_pred[df_pred.day_of_year==df.day_of_year[idx]].values.reshape(-1))
    else:
        print(df['segment_file'][idx], 'missing')

labels = np.stack(labels)
images = np.stack(images)
sup = np.stack(sup)

#%%
# Number of feature variations
all_attr = [col[:-len(sel_model)-6] for col in df_pred.columns if col.__contains__('_w0_') and col.__contains__('_all')]

attr_dict = dict({'mean':all_attr[-1],\
                'distribution':all_attr[-2:], \
                # 'median':all_attr[-7], \
                'quantiles':all_attr[:-2]})

assert attr_dict['mean']=='brix', 'Incorrect attributes read (mean)'
# assert attr_dict['median']=='brix-q50', 'Incorrect attributes read (median)'

#%%
# Number of time windows
all_tw = [col[len(sel_model)+3:-3] for col in df_pred.columns if col.__contains__(sel_model) and col.__contains__('brix_') and col.__contains__('_all')]
assert len(all_tw)==11, 'Incorrect number of time windows' #missing l3-l1

#%%
# Number of models always =1
all_model = [col[4:-6] for col in df_pred.columns if col.__contains__('_w0_') and col.__contains__('brix_') and col.__contains__('_all')]
assert len(all_model)==1, 'Incorrect number of exp-2 models'

#%%
# Number of features
all_fea = [col[len(sel_model)+6:] for col in df_pred.columns if col.__contains__(sel_model) and col.__contains__('brix_') and col.__contains__('_w0_')]
assert len(all_fea)==3, 'Incorrect number of feature groups'

#%%
supScaler = StandardScaler()
sups = supScaler.fit_transform(sup)

# %%
attr_key = 'mean'
tw = all_tw[0]
fea = all_fea[0]
mlp = all_model[0]
attr = all_attr[0]

col_locator = [i for i,col in enumerate(df_pred.columns[12:]) if any(x in col for x in attr_dict[attr_key])]
attr_cols = [col for col in df_pred.columns[12:] if any(x in col for x in attr_dict[attr_key])]
sup_data_attr = sup[:,col_locator]

sub_col_locator_0 = [i for i,col in enumerate(attr_cols) \
                if col.__contains__(tw) and col.__contains__(mlp) \
                and col.__contains__(fea)]
sup_data_0 = sup_data_attr[:,sub_col_locator_0]
im_train_0, _, _ = get_splited_data(np.arange(labels.shape[0]), images, labels, sup_data_0, encoder, sel_pieces=1)

create_models(im_train_0.shape[1]+sup_data_0.shape[1], attr_key, encoder, series=seed)


im_train, y_train, im_test, y_test, im_val, y_val, yScaler, \
sup_train_m, sup_test_m, sup_val_m, i_val \
= generate_dataset_d(images, labels, sup_data_attr, val_per=0.2, test_per=0.15, seed=seed, sel_pieces=1, encoder=encoder)

y_val_beforeScale = yScaler.inverse_transform(y_val.reshape(-1,1))

title = f'by-{encoder}-{attr_key}-{mlp}-by-{tw[1:-1]}-{sel_model[1:-1]}-{fea[1:]}'

model = keras.models.load_model(f'models/{attr_key}_r_empty_fc{mlp[-1]}_{encoder}_{seed_fraction}.h5', compile=False)
mc = ModelCheckpoint(f'models/reg_{title}_ckpt_s{seed_fraction}.h5', monitor='val_loss', mode='min',\
    verbose=0, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0003), loss='mse')

sub_col_locator = [i for i,col in enumerate(attr_cols) \
            if col.__contains__(tw) and col.__contains__(sel_model) \
            and col.__contains__(fea)]

X_train = np.hstack((im_train, sup_train_m[:,sub_col_locator]))
X_test = np.hstack((im_test, sup_test_m[:,sub_col_locator]))
X_val = np.hstack((im_val, sup_val_m[:,sub_col_locator]))

history = model.fit(x=X_train, y=y_train, epochs=epoch, batch_size=16, \
    validation_data=(X_test, y_test), callbacks=[es, mc], verbose=0)

y_pred = yScaler.inverse_transform(model.predict(X_val))
err = y_val_beforeScale.reshape(-1)-y_pred.reshape(-1)

err_df = pd.DataFrame(np.repeat(i_val, sel_pieces=1).astype(int))
err_df[['err']] = err.reshape(-1,1)
err_df.to_csv(f'results/pred_errors_{key}.csv')