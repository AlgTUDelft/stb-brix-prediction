#%%
# Basic Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import scipy
from scipy.stats import norm

from sklearn.metrics import mean_squared_error
from math import sqrt

# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%%
# Random Seed Lists
import random
random.seed(2022)
randomlist15 = random.sample(range(3000), 15)

#%%
# General Functions
def rmse(arr):
    return np.sqrt(np.dot(arr,arr)/len(arr))

def model_eval(model, X_test, y_test, scalerY):
    y_pred = model.predict(X_test).reshape(-1,1)

    y_test_real = scalerY.inverse_transform(y_test).reshape(-1)
    y_pred_real = scalerY.inverse_transform(y_pred).reshape(-1)
    err = y_pred_real-y_test_real
    rmse = sqrt(mean_squared_error(y_pred_real, y_test_real))

    mu, sigma = norm.fit(err)

    score = model.score(X_test,y_test)

    return err, rmse, mu, sigma, score

def err_plot(model, X_test, y_test, scalerY, dataset_size=100, model_name='Pred_Model', seed=0, \
    bins = np.linspace(-10,10,41), plot=False, save_err=True):
    err, rmse, mu, sigma, score = model_eval(model, X_test, y_test, scalerY)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

    if plot:
        plt.close('all')
        plt.figure()
        plt.hist(err, bins)
        plt.title('Error Distribution of '+model_name+ '\n' + \
                    'err_mean='+str(round(mu,2))+ \
                    ', rmse='+str(round(rmse,2))+ \
                    ', err_std='+str(round(sigma,2))+ \
                    ', R2 score='+str(round(score,2)))
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.xlim(-3,3)
        # plt.ylim(0,120)
        
        plt.plot(bins, best_fit_line*dataset_size)

        # plt.legend('err_mean='+str(round(mu,2))+ \
        #             ', rmse'+str(round(rmse,1))+ \
        #             ', err_std='+str(round(sigma,1))+ \
        #             ', R2 score='+str(round(score,1)))

        plt.tight_layout()
        plt.savefig(model_name.replace(' ', '_')+f'_output_s{seed}.png')
    
    if save_err:
        pd.DataFrame(err).to_csv(model_name.replace(' ', '_')+f'_err_s{seed}.csv')

    return mu, rmse, sigma

#%% Random Splits
def split_and_scale(X, y, s=123, ts=0.25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=s)

    scalerX = StandardScaler().fit(X_train)
    scalerY = StandardScaler().fit(y_train)

    X_train = scalerX.transform(X_train)
    y_train = scalerY.transform(y_train.reshape(-1, 1))

    X_test = scalerX.transform(X_test)
    y_test = scalerY.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test, scalerY


def LR(df, attr, info, info_2_col, fea_group='', seeds=[123], save=False, save_best=0):
    sel = info_2_col[info]
    X = df[df.columns[sel.astype(int)]].values
    y = df.values[:,-1].reshape(-1,1)

    mus=[]
    rmses=[]
    sigmas=[]
    scores=[]
    all_weights=[]

    if save_best>0:
        best_r2 = -1*np.ones(int(save_best))

    for s in seeds:
        X_train, X_test, y_train, y_test, scalerY = split_and_scale(X,y,s)
        reg = LinearRegression().fit(X_train, y_train)

        if s!=seeds[-1]:
            mu, rmse, sigma = err_plot(reg, X_test, y_test, scalerY, df.shape[0], f'{attr}_LR_with_{info}_{fea_group}', seed=s)
        else:
            mu, rmse, sigma = err_plot(reg, X_test, y_test, scalerY, df.shape[0], f'{attr}_LR_with_{info}_{fea_group}', plot=True, seed=s)
        
        score = reg.score(X_test,y_test)

        mus = np.append(mus,mu)
        rmses = np.append(rmses,rmse)
        sigmas = np.append(sigmas,sigma)
        scores = np.append(scores,score)

        weights=reg.coef_.round(2)
        all_weights = np.append(all_weights,weights)

        # save the model
        if save_best>0 and score>best_r2[0]:
            best_r2[0] = score
            best_r2 = np.sort(best_r2)
            filename = f'{attr}_LR_{info}_{fea_group}_b{best_r2.index(score)}.sav'
            pickle.dump(reg, open(filename, 'wb'))
        
        # # load the model from disk
        # loaded_model = pickle.load(open(filename, 'rb'))
        # result = loaded_model.score(X_test, Y_test)
        # print(result)

    # print('median performance of {} with {} and LR:'.format(attr,info), np.median(scores).round(2), np.median(rmses).round(2))
         
    all_weights = pd.DataFrame(all_weights.reshape(len(seeds),-1).round(2),columns=df.columns[sel.astype(int)])
    if save:
        all_weights.to_csv(f'{attr}_LR_weights_{info}_{fea_group}.csv')

    # return [mus, rmses, sigmas, scores], all_weights
    return [np.median(mus), np.median(rmses), np.median(sigmas), np.median(scores)], all_weights

def KRR(df, attr, info, info_2_col, d=1, a=1, fea_group='', seeds=[123], save=False, save_best=0):
    sel = info_2_col[info]
    # print(sel)
    X = df[df.columns[sel.astype(int)]].values
    y = df.values[:,-1].reshape(-1,1)

    mus=[]
    rmses=[]
    sigmas=[]
    scores=[]
    all_weights=[]
    
    for s in seeds:
        X_train, X_test, y_train, y_test, scalerY = split_and_scale(X,y,s)

        if d!=1:
            reg = KernelRidge(kernel='poly',alpha=a,degree=d).fit(X_train, y_train)
        else:
            reg = Ridge(alpha=a).fit(X_train, y_train)
            weights=reg.coef_
            all_weights = np.append(all_weights,weights)

        if s!=seeds[-1]:
            mu, rmse, sigma = err_plot(reg, X_test, y_test, scalerY, df.shape[0], f'{attr}_KRR-a{a}-d{d}_with_{info}_{fea_group}', seed=s)
        else:
            mu, rmse, sigma = err_plot(reg, X_test, y_test, scalerY, df.shape[0], f'{attr}_KRR-a{a}-d{d}_with_{info}_{fea_group}', plot=True, seed=s)
        
        score = reg.score(X_test,y_test)

        mus = np.append(mus,mu)
        rmses = np.append(rmses,rmse)
        sigmas = np.append(sigmas,sigma)
        scores = np.append(scores,score)

        # save the model
        if save_best>0 and score>best_r2[0]:
            best_r2[0] = score
            best_r2 = np.sort(best_r2)
            filename = f'{attr}_KRR-a{a}-d{d}_b{best_r2.index(score)}_{info}_{fea_group}.sav'
            pickle.dump(reg, open(filename, 'wb'))

    # print('average performance:', np.mean(score).round(2), np.mean(rmse).round(2))

    if d==1:
        all_weights = pd.DataFrame(all_weights.reshape(len(seeds),-1),columns=df.columns[sel.astype(int)])
        if save:
            all_weights.to_csv(f'{attr}_KRR-a{a}-d{d}_weights_{info}_{fea_group}.csv')

    return [mu, rmse, sigma, reg.score(X_test,y_test).round(2)], all_weights
    

def SVRtrain(df, attr, info, info_2_col, k='rbf', fea_group='', seeds=[123], save=False, save_best=0):
    sel = info_2_col[info]
    X = df[df.columns[sel.astype(int)]].values
    y = df.values[:,-1].reshape(-1,1)

    mus=[]
    rmses=[]
    sigmas=[]
    scores=[]
    all_weights=[]

    if save_best>0:
        best_r2 = -1*np.ones(int(save_best))

    for s in seeds:
        X_train, X_test, y_train, y_test, scalerY = split_and_scale(X,y,s)

        if k.__contains__('poly'):
            reg = SVR(kernel='poly',degree=int(k[-1])).fit(X_train, y_train.ravel())
        else:
            reg = SVR(kernel=k).fit(X_train, y_train.ravel())

        if s!=seeds[-1]:
            mu, rmse, sigma = err_plot(reg, X_test, y_test, scalerY, df.shape[0], f'{attr}_SVR-{k}_with_{info}_{fea_group}', seed=s)
        else:
            mu, rmse, sigma = err_plot(reg, X_test, y_test, scalerY, df.shape[0], f'{attr}_SVR-{k}_with_{info}_{fea_group}', plot=True, seed=s)
 
        score = reg.score(X_test,y_test)

        mus = np.append(mus,mu)
        rmses = np.append(rmses,rmse)
        sigmas = np.append(sigmas,sigma)
        scores = np.append(scores,score)

        # weights=reg.coef_.round(2)
        # all_weights = np.append(all_weights,weights)

        # save the model
        if save_best>0 and score>best_r2[0]:
            best_r2[0] = score
            best_r2 = np.sort(best_r2)
            filename = f'SVR-{k}_{attr}_b{best_r2.index(score)}_{info}_{fea_group}.sav'
            pickle.dump(reg, open(filename, 'wb'))
        
    # print('median performance of {} with {} and LR:'.format(attr,info), np.median(scores).round(2), np.median(rmses).round(2))
         
    # all_weights = pd.DataFrame(all_weights.reshape(len(seeds),-1).round(2),columns=df.columns[sel])
    # if save:
    #     all_weights.to_csv('weights_LR_{}.csv'.format(info))

    return [np.median(mus), np.median(rmses), np.median(sigmas), np.median(scores)], all_weights    


def run_lr(df,attr,info_2_col,fea_group=''):
    result_summary = pd.DataFrame()
    all_weights = pd.DataFrame(columns=df.columns) 
    for info in info_2_col.keys():
        results_table = pd.DataFrame()
        [mu, rmse, sigma, score], weights = LR(df, attr, info, info_2_col, fea_group=fea_group, save=True, seeds=randomlist15)
        result = pd.DataFrame([mu, rmse, sigma, score]).transpose()
        results_table = pd.concat([results_table,result])
        results_table.columns=[['mu','rmse','sigma','score']]
        results_table.to_csv(f'LR_results_{info}_{attr}_{fea_group}.csv')
        result_summary[[info]] = pd.DataFrame(results_table.mean())

        all_weights = pd.concat([all_weights,weights.mean(axis=0).to_frame().T])

    return result_summary, all_weights


def run_kr(df,attr,info_2_col,a=1,d=1,fea_group=''):
    result_summary = pd.DataFrame()
    all_weights = pd.DataFrame(columns=df.columns) 

    # for dd in range(1,4):
    for info in info_2_col.keys():
        # print(dd,info)
        results_table = pd.DataFrame()
        [mu, rmse, sigma, score], weights = KRR(df, attr, info, info_2_col, d=d, a=a, fea_group=fea_group, save=True, seeds=randomlist15)
        result = pd.DataFrame([mu, rmse, sigma, score]).transpose()
        results_table = pd.concat([results_table,result])
        results_table.columns=[['mu','rmse','sigma','score']]
        # results_table.columns=[col + '-a{}'.format(dd) for col in ['mu','rmse','sigma','score']]
        results_table.to_csv(f'KRR-a{a}-d{d}_results_{info}_{attr}_{fea_group}.csv')
        result_summary[[info]] = pd.DataFrame(results_table.mean())

        if d==1:
            all_weights = pd.concat([all_weights,weights.mean(axis=0).to_frame().T])

    return result_summary, all_weights


def run_svr(df,attr,info_2_col,k='rbf',fea_group=''):
    result_summary = pd.DataFrame()
    all_weights = [] #pd.DataFrame(columns=df.columns) 
    for info in info_2_col.keys():
        results_table = pd.DataFrame()
        [mu, rmse, sigma, score], weights = SVRtrain(df, attr, info, info_2_col, fea_group=fea_group, save=True, seeds=randomlist15, k=k)
        result = pd.DataFrame([mu, rmse, sigma, score]).transpose()
        results_table = pd.concat([results_table,result])
        results_table.columns=[['mu','rmse','sigma','score']]
        results_table.to_csv(f'SVR-{k}_results_{info}_{attr}_{fea_group}.csv')
        result_summary[[info]] = pd.DataFrame(results_table.mean())

    return result_summary, all_weights


#%%
# Leave-ont-out Splits

def leave_one_and_scale(X, y, exclude=0):
    X_test = X[exclude,:].reshape(1,-1)
    y_test = y[exclude,:].reshape(1,-1)
    X_train = np.delete(X, exclude, 0)
    y_train = np.delete(y, exclude, 0)
        
    scalerX = StandardScaler().fit(X_train)
    scalerY = StandardScaler().fit(y_train)

    X_train = scalerX.transform(X_train)
    y_train = scalerY.transform(y_train.reshape(-1, 1))

    X_test = scalerX.transform(X_test)
    y_test = scalerY.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test, scalerY


def LR_e(df, info, info_2_col, exclude=0):
    # model = 'LR'

    sel = info_2_col[info]
    X = df[df.columns[sel.astype(int)]].values
    y = df.values[:,-1].reshape(-1,1)

    X_train, X_test, y_train, y_test, scalerY = leave_one_and_scale(X, y, exclude)
    y_test = scalerY.inverse_transform(y_test).reshape(-1,1)

    reg = LinearRegression().fit(X_train, y_train)
    score = reg.score(X_train,y_train)

    y_pred = scalerY.inverse_transform(reg.predict(X_test).reshape(-1,1))
    err = y_test-y_pred

    weights = reg.coef_
  
    # print('fitting performance (R2) of {} with {} and {}:'.format(attr,info,model), score.round(2))

    return err, score, weights
    # return [np.median(mus), np.median(rmses), np.median(sigmas), np.median(scores)], all_weights


def KR_e(df, info, info_2_col, exclude=0, a=1, d=1):
    # model = 'KRR'

    sel = info_2_col[info]
    X = df[df.columns[sel.astype(int)]].values
    y = df.values[:,-1].reshape(-1,1)

    X_train, X_test, y_train, y_test, scalerY = leave_one_and_scale(X, y, exclude)
    y_test = scalerY.inverse_transform(y_test).reshape(-1,1)

    if d!=1:
        reg = KernelRidge(kernel='poly',alpha=a,degree=d).fit(X_train, y_train)
    else:
        reg = Ridge(alpha=a).fit(X_train, y_train)

    score = reg.score(X_train,y_train)

    y_pred = scalerY.inverse_transform(reg.predict(X_test).reshape(-1,1))
    err = y_test-y_pred

    if d==1: weights = reg.coef_
    else: weights=[]
  
    # print('fitting performance (R2) of {} with {} and {}:'.format(attr,info,model), score.round(2))

    return err, score, weights
    # return [np.median(mus), np.median(rmses), np.median(sigmas), np.median(scores)], all_weights


def SVR_e(df, info, info_2_col, exclude=0, k='rbf'):
    # model = 'SVR_'+k

    sel = info_2_col[info]
    X = df[df.columns[sel.astype(int)]].values
    y = df.values[:,-1].reshape(-1,1)

    X_train, X_test, y_train, y_test, scalerY = leave_one_and_scale(X, y, exclude)
    y_test = scalerY.inverse_transform(y_test).reshape(-1,1)
    
    if k.__contains__('poly'):
        reg = SVR(kernel='poly',degree=int(k[-1])).fit(X_train, y_train.ravel())
    else:
        reg = SVR(kernel=k).fit(X_train, y_train.ravel())
        
    score = reg.score(X_train,y_train)

    y_pred = scalerY.inverse_transform(reg.predict(X_test).reshape(-1,1))
    err = y_test-y_pred

    weights=[]
    # weights = reg.coef_
  
    # print('fitting performance (R2) of {} with {} and {}:'.format(attr,info,model), score.round(2))

    return err, score, weights
    # return [np.median(mus), np.median(rmses), np.median(sigmas), np.median(scores)], all_weights


def run_lre(df,attr,info_2_col,fea_group=''):
    m='LR'
    all_weights = pd.DataFrame(columns=df.columns)
    result_summary = pd.DataFrame()
    for info in info_2_col.keys():
        results_table = pd.DataFrame(columns=[['err','score']])
        # results_table
        for i in range(len(df)):
            err, score, weights = LR_e(df, info, info_2_col, exclude=i)
            result = pd.DataFrame([err[0,0], score]).transpose()
            result.columns=[['err','score']]
            results_table = pd.concat([results_table, result])
            df_weights = pd.DataFrame(weights, columns=df.columns[info_2_col[info].astype(int)])
            all_weights = pd.concat([all_weights,df_weights])
        
        results_table.to_csv(f'{m}e_results_{info}_{attr}_{fea_group}.csv')
        all_weights.to_csv(f'{m}e_weights_{info}_{attr}_{fea_group}.csv')

        # result_summary[[info]] = pd.DataFrame(results_table.mean())
        re = rmse(results_table[['err']].values.reshape(-1))
        result_summary[[info]] = pd.Series([list(results_table.mean())+[re]])

    result_summary.to_csv(f'{m}e_avg-results_{info}_{attr}_{fea_group}.csv')

    return result_summary


def run_kre(df,attr,info_2_col,a=1,d=1,fea_group=''):
    m='KRR'
    
    all_weights = pd.DataFrame(columns=df.columns) 

    # for dd in range(1,4):
    result_summary = pd.DataFrame()
    for info in info_2_col.keys():
        # print(dd,info)
        results_table = pd.DataFrame(columns=[['err','score']])
        # results_table
        for i in range(len(df)):
            err, score, weights = KR_e(df, info, info_2_col, exclude=i,a=a)
            result = pd.DataFrame([err[0,0], score]).transpose()
            result.columns=[['err','score']]
            results_table = pd.concat([results_table, result])
            
            if d==1: 
                df_weights = pd.DataFrame(weights, columns=df.columns[info_2_col[info].astype(int)])
                all_weights = pd.concat([all_weights,df_weights])
        
        results_table.to_csv(f'{m}e-a{a}-d{d}_results_{info}_{attr}_{fea_group}.csv')
        
        if d==1:
            all_weights.to_csv(f'{m}e-a{a}-d{d}_weights_{info}_{attr}_{fea_group}.csv')

        # result_summary[[info]] = pd.DataFrame(results_table.mean())
        re = rmse(results_table[['err']].values.reshape(-1))
        result_summary[[info]] = pd.Series([list(results_table.mean())+[re]])

    result_summary.to_csv(f'{m}e-a{a}-d{d}_avg-results_{info}_{attr}_{fea_group}.csv')

    return result_summary


def run_svre(df,attr,info_2_col,k='rbf',fea_group=''):
    m='SVR'
    # all_weights = pd.DataFrame(columns=df.columns)
    result_summary = pd.DataFrame()
    for info in info_2_col.keys():
        results_table = pd.DataFrame(columns=[['err','score']])
        # results_table
        for i in range(len(df)):
            err, score, _ = SVR_e(df, info, info_2_col,exclude=i,k=k)
            result = pd.DataFrame([err[0,0], score]).transpose()
            result.columns=[['err','score']]
            results_table = pd.concat([results_table, result])

            # df_weights = pd.DataFrame(weights, columns=df.columns[info_2_col[info]])
            # all_weights = pd.concat([all_weights,df_weights])
        
        results_table.to_csv(f'{m}e-{k}_results_{info}_{attr}_{fea_group}.csv')
        # all_weights.to_csv('{}e_weights_{}_{}.csv'.format(m, info, attr))

        # result_summary[[info]] = pd.DataFrame(results_table.mean())
        re = rmse(results_table[['err']].values.reshape(-1))
        result_summary[[info]] = pd.Series([list(results_table.mean())+[re]])

    result_summary.to_csv(f'{m}e-{k}_avg-results_{info}_{attr}_{fea_group}.csv')
    return result_summary