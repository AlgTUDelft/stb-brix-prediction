from pathlib import Path
import os
from glob import glob

import pandas as pd
import numpy as np
import copy
import datetime

import random

from utils.reg_utils import *

basePath = Path(os.getcwd()).parent
dataPath = os.path.join(basePath, 'example_data')

attr = 'brix'
key = f'{attr}-expb-demo'

seed = 2022
random.seed(seed)

# data
env_data = pd.read_csv(os.path.join(dataPath,'env_rolling.csv'), index_col=0)
mea_data = pd.read_csv(os.path.join(dataPath,'Strawberry_Measurements_with_Seg_Connections_mtd1.csv'), index_col=0)

cols = env_data.columns.tolist()

if 'week_of_year' in cols:
    env_data = env_data.drop(columns='week_of_year')
    
#value check
mea_data['Strawberry ID'] = mea_data['Strawberry ID'].str.replace('.',',')
mea_data['Strawberry ID'] = mea_data['Strawberry ID'].str.replace('4,28','28,4')

mea_data = mea_data.dropna(subset=['Strawberry ID',attr])

mea_data[['dd','mm','br','nr']]=mea_data['Strawberry ID'].str.split(",",expand=True,)
mea_data[['dd','mm']] = mea_data[['dd','mm']].astype(int)
mea_data[['yy']] = (np.ones(shape=(mea_data.shape[0],1))*2021).astype(int)
mea_data['date'] = mea_data.apply(lambda x: datetime.date(x['yy'], x['mm'], x['dd']), axis=1)
mea_data['formatted_date'] = pd.to_datetime(mea_data['date'])
mea_data['day_of_year'] = mea_data.formatted_date.apply(lambda x: x.dayofyear)
mea_data['week_of_year'] = mea_data.formatted_date.apply(lambda x: x.weekofyear)

dfb = mea_data[['day_of_year', attr]].dropna(subset=[attr])

# select features
fea_group='all'

cols = ['day_of_year'] + [col for col in env_data.columns if col!='day_of_year']
env_w = copy.copy(env_data[cols])

n_env = (env_w.shape[1]-1)/5

info_2_col = {}
info_2_col['l3w'] = np.arange(1,n_env*1+1)
info_2_col['l2w'] = np.arange(n_env*1+1,n_env*2+1)
info_2_col['l1w'] = np.arange(n_env*2+1,n_env*3+1)
info_2_col['w0'] = np.arange(n_env*3+1,n_env*4+1)

info_2_col['f1w'] = np.arange(n_env*4+1,n_env*5+1)

# generate combinations
info_2_col_env_combi = {}
info_2_col_env_combi['w0'] = info_2_col['w0']
info_2_col_env_combi['l1-w0'] = np.append(info_2_col['l1w'],info_2_col['w0'])
info_2_col_env_combi['l2-w0'] = np.append(info_2_col['l2w'],info_2_col_env_combi['l1-w0'])
info_2_col_env_combi['l3-w0'] = np.append(info_2_col['l3w'],info_2_col_env_combi['l2-w0'])

info_2_col_env_combi['l1w'] = info_2_col['l1w']
info_2_col_env_combi['l2-l1'] = np.append(info_2_col['l2w'],info_2_col['l1w'])
info_2_col_env_combi['l3-l1'] = np.append(info_2_col['l3w'],info_2_col_env_combi['l2-l1'])

info_2_col_env_combi['l2w'] = info_2_col['l2w']
info_2_col_env_combi['l3-l2'] = np.append(info_2_col['l3w'],info_2_col['l2w'])

info_2_col_env_combi['w0-f1'] = np.append(info_2_col['w0'],info_2_col['f1w'])
info_2_col_env_combi['l1-f1'] = np.append(info_2_col['l1w'],info_2_col_env_combi['w0-f1'])

df2e = dfb.groupby('day_of_year')[attr].mean().reset_index()

df2e = df2e.merge(env_w, on=['day_of_year'])
df2e = df2e[[c for c in df2e if c.__contains__('year')] + [c for c in df2e if c not in ['year',attr]] + [attr]]
df2e = df2e.loc[:,~df2e.columns.duplicated()]

new_col = [col for col in df2e.columns if not col.__contains__('=2')]
df2e = df2e[new_col]

# run
a = 100
d = 1
result_summary, all_weights = run_kr(df2e, attr, info_2_col_env_combi, a=a, d=d, fea_group=fea_group)