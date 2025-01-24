from pathlib import Path
import os
from glob import glob
from pickletools import int4
from timeit import repeat

import pandas as pd
import numpy as np
import copy
import json

from PIL import Image

import random
import datetime

from utils.utils import *
from utils.models import *

basePath = Path(os.getcwd()).parent
dataPath = os.path.join(basePath, 'example_data')

# Parameters
attr = 'brix'
key = f'{attr}-expa-demo'

epochs = 200
seed = 2022
batch_size = 16
model_num = '0'

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

images = []
labels = []

for idx in df.index:
    img_file = os.path.join(dataPath, 'Segments', df['segment_file'][idx])
    if os.path.exists(img_file):
        im = Image.open(img_file)
        im_array = np.array(im)
        im_array[im_array[:,:,3]==0]=0
        images.append(im_array)
        labels.append(df['Sugariness'][idx])
    else:
        print(df['segment_file'][idx], 'missing')

labels = np.stack(labels)
images = np.stack(images)


# load pre-trained encoder
encoder = keras.models.load_model(os.path.join(dataPath, 'example_encoder.h5'))
im_train, _ = get_data_ae(np.arange(labels.shape[0]), images, labels, encoder=encoder)

im_train, y_train, im_val, y_val, im_test, y_test, yScaler \
= generate_dataset(images, labels, val_per=0.2, test_per=0, seed=seed, encoder=encoder)

pca = PCA(n_components = im_train.shape[1]).fit(im_train.reshape(im_train.shape[0],-1))

im_train = pca.transform(im_train.reshape(im_train.shape[0],-1))
im_test = pca.transform(im_test.reshape(im_test.shape[0],-1))
im_val = pca.transform(im_val.reshape(im_val.shape[0],-1))

ytest_beforeScale = yScaler.inverse_transform(y_test.reshape(-1,1))
bins = np.linspace(-5,5,11)

X_train = im_train
X_test = im_test
X_val = im_val

model = keras.models.load_model(f'models/empty_mlp_{model_num}.h5')
mc = ModelCheckpoint(f'models/trained_expa_{key}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0003), loss='mse')

history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[es, mc])

y_pred = yScaler.inverse_transform(model.predict(X_test))
errors = ytest_beforeScale.reshape(-1)-y_pred.reshape(-1)

with open(f'results/pred_errors_{key}.json', 'w') as fp:
    json.dump(errors, fp)