__author__ = 'kevin'

import numpy as np
from sklearn import linear_model

# parameters
ndata = 300
ndim = 100
ndim_meaningful = 4

y_noise_std = 4
betas_std = 3

alpha = .3  # regularization coefficient

seed = 2

# basic setup
ndim_noise = ndim - ndim_meaningful
np.random.seed(seed)

def l2_error(y_true, y_pred):
    diff = (y_true-y_pred)
    return np.sqrt(np.dot(diff, diff)) / len(diff)


# create beta, and training x and y
# beta_true = np.random.randn(ndim_meaningful) * betas_std
beta_true = np.arange(ndim_meaningful) + 1
x_core_train = np.random.randn(ndata, ndim_meaningful)
x_noise_train = np.random.randn(ndata, ndim_noise)
x_full_train = np.hstack([x_core_train, x_noise_train])

y_true_train = np.dot(x_core_train, beta_true) + np.random.randn(ndata) * y_noise_std

# create test set data
x_core_test = np.random.randn(ndata, ndim_meaningful)
x_noise_test = np.random.randn(ndata, ndim_noise)
x_full_test = np.hstack([x_core_test, x_noise_test])

y_true_test = np.dot(x_core_test, beta_true) + np.random.randn(ndata) * y_noise_std


# fit ordinary least squares model
lm = linear_model.LinearRegression()
lm.fit(x_full_train, y_true_train)
y_lm = lm.predict(x_full_train)

# fit l1 penalized OLS model
lm1 = linear_model.Lasso(alpha=alpha)
lm1.fit(x_full_train, y_true_train)
y_lm1 = lm1.predict(x_full_train)

# fit l2 penalized OLS model
lm2 = linear_model.Ridge(alpha=alpha)
lm2.fit(x_full_train, y_true_train)
y_lm2 = lm2.predict(x_full_train)


# # display parameters
# n_padding_coefs = 2 # number of non-meaningful coefs to display
# n_data_disp = 10

# n_coefs_disp = ndim_meaningful + n_padding_coefs

# # display true/learned coefs
# beta_true_disp = np.concatenate([beta_true, np.zeros(n_padding_coefs)])
# beta_lm_disp = lm.coef_[:n_coefs_disp]
# beta_lm1_disp = lm1.coef_[:n_coefs_disp]
# beta_lm2_disp = lm2.coef_[:n_coefs_disp]
#
# betas = np.hstack([beta_true_disp[:, np.newaxis],
#                    beta_lm_disp[:, np.newaxis],
#                    beta_lm1_disp[:, np.newaxis],
#                    beta_lm2_disp[:, np.newaxis],
#                    ])
# print "BETAS:"
# print '  % 11s % 11s % 11s % 11s' % ('true', 'lm', 'lm1', 'lm2')
# print betas
#
# # display errors
# print
# print "Train / Test ERRORS:"
# print '  % 11s % 11s % 11s % 11s' % ('true', 'lm', 'lm1', 'lm2')
# print '  % 11f % 11f % 11f % 11f' % (0,
#                                      l2_error(y_true_train, y_lm),
#                                      l2_error(y_true_train, y_lm1),
#                                      l2_error(y_true_train, y_lm2),)
#
#
# # display predictions
# preds = np.hstack([y_true_train[:, np.newaxis],
#                    y_lm[:, np.newaxis],
#                    y_lm1[:, np.newaxis],
#                    y_lm2[:, np.newaxis]])

# print
# print "PREDS:"
# print '  % 11s % 11s % 11s % 11s' % ('true', 'lm', 'lm1', 'lm2')
# print preds[:n_data_disp]

def display_results(models, model_names, betas_true, x_train, y_train, x_test, y_test):
    n_padding_coefs = 3
    n_dims_disp = len(betas_true) + n_padding_coefs

    # fit models to training data
    [m.fit(x_train, y_train) for m in models]

    # display learned betas
    betas = [m.coef_[:n_dims_disp, np.newaxis] for m in models]
    alphas = [m.alpha for m in models]
    print "ALPHAS:"
    print '  % 11s % 11s % 11s % 11s' % ('true', 'lm', 'lm1', 'lm2')
    print alphas

    betas_true_disp = np.concatenate([beta_true, np.zeros(n_padding_coefs)])[:, np.newaxis]
    betas =  [betas_true_disp] + betas
    print "BETAS:"
    print '  % 11s % 11s % 11s % 11s' % ('true', 'lm', 'lm1', 'lm2')
    print np.hstack(betas)

    # display mean squared error (MSE) for test data
    y_preds = [m.predict(x_test) for m in models]
    errors = [l2_error(y_test, y_pred) for y_pred in y_preds]
    errors = tuple([0] + errors)
    print
    print "Test Errors:"
    print '  % 11s % 11s % 11s % 11s' % ('true', 'lm', 'lm1', 'lm2')
    print '  % 11f % 11f % 11f % 11f' % errors

    # display predictions

models = [lm, lm1, lm2]
display_results(models, [], beta_true, x_full_train, y_true_train, x_full_test, y_true_test)
