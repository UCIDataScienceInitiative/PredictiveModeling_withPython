from __future__ import division
__author__ = 'kevin'

import numpy as np
import pandas as pd
from sklearn import linear_model

# parameters
ndata_train = 600
ndata_test = 300

ndim = 598
ndim_meaningful = 3
ndim_disp_padding = 2

y_noise_std = 1

seed = 3

# basic setup
np.random.seed(seed)
ndim_noise = ndim - ndim_meaningful

def mean_squared_error(y_true, y_pred):
    diff = y_true - y_pred
    return np.dot(diff, diff) / len(y_true)

def model_name(model):
    s = model.__str__().lower()
    if "linearregression" in s:
        return 'LinearRegression'
    elif "lasso" in s:
        return 'Lasso(a=%g)' % model.alpha
    elif "ridge" in s:
        return 'Ridge(a=%g)' % model.alpha
    elif "elastic" in s:
        return 'ElasticNet(a=%g, r=%g)' % (model.alpha, model.l1_ratio)
    else:
        raise ValueError("Unknown Model Type")

def results_df(models, betas_true, x_train, y_train, x_test, y_test):
    n_zeros = ndim - len(betas_true)
    betas_true = np.concatenate([betas_true, np.zeros(n_zeros)])

    # fit models to training data
    [m.fit(x_train, y_train) for m in models]

    betas = np.vstack([betas_true] + [m.coef_ for m in models])
    beta_names = ['Beta ' + str(i) for i in range(ndim)]

    # set up model names
    model_names =  ["True Coefs"] + [model_name(m) for m in models]
    df = pd.DataFrame(data=betas, columns=beta_names, index=model_names)

    y_preds = [m.predict(x_train) for m in models]
    errors = [np.nan] + [mean_squared_error(y_train, y_pred) for y_pred in y_preds]
    df['Train Error'] = errors

    y_preds = [m.predict(x_test) for m in models]
    errors = [np.nan] + [mean_squared_error(y_test, y_pred) for y_pred in y_preds]
    df['Test Error'] = errors

    return df


def create_models(alphas=(.01, .03, .1, .3, 1, 3), l1_ratios=(.7, .5, .3)):
    models = [linear_model.LinearRegression()]
    models.extend([linear_model.Ridge(a) for a in alphas])
    models.extend([linear_model.Lasso(a) for a in alphas])
    models.extend([linear_model.ElasticNet(a, l1_ratio=l) for a in alphas for l in l1_ratios])
    return models


# create true betas
beta_true = np.arange(ndim_meaningful) + 1

# # create train data
# x_core_train = np.random.randn(ndata_train, ndim_meaningful)
# x_full_train = np.hstack([x_core_train, x_noise_train])
# y_true_train = np.dot(x_core_train, beta_true) + np.random.randn(ndata_train) * y_noise_std
#
# # create test data
# x_core_test = np.random.randn(ndata_train, ndim_meaningful)
# x_noise_test = np.random.randn(ndata_train, ndim_noise)
# x_full_test = np.hstack([x_core_test, x_noise_test])
# y_true_test = np.dot(x_core_test, beta_true) + np.random.randn(ndata_train) * y_noise_std

# create full dataset
ndata_total = ndata_train + ndata_test
x_core = np.random.randn(ndata_total, ndim_meaningful)
x_noise = np.random.randn(ndata_total, ndim_noise)
X = np.hstack([x_core, x_noise])
y = np.dot(x_core, beta_true) + np.random.randn(ndata_total) * y_noise_std

from sklearn.cross_validation import train_test_split
test_size = ndata_test/ndata_total
print test_size
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

print x_train.shape
print x_test.shape

# save temporalized version of data
# np.savez('user_data_old.npz', user_data_yesterday=x_train, rating_yesterday=y_train, user_data_tomorrow=x_test)
# np.savez('user_data_old.npz', rating_tomorrow=y_test)
np.savez('overfitting_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

models = create_models()
df = results_df(models, beta_true, x_train, y_train, x_test, y_test)
disp_cols = ["Beta " + str(i) for i in range(ndim_meaningful + ndim_disp_padding)] + ['Train Error', 'Test Error']
# print df.ix[:, [0, 1, 2, 3, 100, 101]]
print df[disp_cols]

