import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import urllib.request
#import urllib
import time
import os
import csv
import numpy.matlib

class finance_basic_info:
    """______________________________________________________________________________________________________________"""
    "______________________________________________Fetch Data and Basic Info___________________________________________"
    def __init__(self, company_list, risk_free= 1/100):
        self.company_list = company_list
        self.risk_free = risk_free

    # checked
    def pullData(self, length=2):
        current_date = time.strftime("%d/%m/%Y")
        current_day = current_date[:2]
        current_mon = current_date[3:5]
        current_year = int(current_date[6:])
        back_year = str(current_year - length)
        #print (back_year)

        for stock in self.company_list:
            fileLine = stock + '.csv'
            urltovisit = 'http://ichart.finance.yahoo.com/table.csv?s=' + stock + '&a=' + current_mon + \
                                        '&b=' + current_day + '&c=' + back_year + '&d=' + current_mon + '&e=' + current_day + \
                                        '&f=' + str(current_year) + '&g=d&ignore=.csv'
            try:
                os.remove(fileLine)
            except OSError:
                pass

            urllib.request.urlretrieve(urltovisit, fileLine)

            print('Pulled', stock)
            print('...')
            time.sleep(.5)

    # checked
    def read_data(self, filename):

        yahoo_finance = pd.read_csv(filename, index_col="Date",
                                          parse_dates=True, usecols=['Date', 'Adj Close'],
                                          na_values=['nan'])
        return yahoo_finance

    # get_data checked
    def get_data(self, start_date, end_date, norm_plot = False):
        date_list = pd.date_range(start_date, end_date)
        df1 = pd.DataFrame(index=date_list)
        company_list = []
        for company in self.company_list:
            company_list.append(company + '.csv')

        for item in company_list:
            temp_dataframe = self.read_data(item)
            temp_dataframe = temp_dataframe.rename(columns={'Adj Close': item[:-4]})
            df1 = df1.join(temp_dataframe, how='inner')
        df1.fillna(method='ffill', inplace=True)
        df1.fillna(method='bfill', inplace=True)
        row_data = df1
        if norm_plot is True:
            plot_data = row_data
            plot_data = plot_data/plot_data.ix[0,:]
            plot_data.plot()
            plt.show()
        return row_data

    # checked
    def rolling_stats(self, dataframe, window = 20):
        r = dataframe.rolling(window = window)
        rollingmean = r.mean()
        rollingstd = r.std()
        upper_band = rollingmean + 2*rollingstd
        lower_band = rollingmean - 2*rollingstd
        return rollingmean, rollingstd, upper_band, lower_band

    # checked
    def dailyret_stats(self, dataframe):
        dailyret = dataframe.copy()
        dailyret [1:] = (dataframe[1:]/dataframe[:-1].values) - 1
        dailyret.ix[0,:] = 0
        totalday = dailyret.shape[0]
        daily_risk_free = self.risk_free/totalday
        sharperatio = np.mean(dailyret - daily_risk_free)/np.std(dailyret-daily_risk_free)
        meanret = dailyret.mean()
        stdret = dailyret.std()
        kurtret = dailyret.kurtosis()
        covret = dailyret.cov()
        return dailyret, meanret, stdret, covret, kurtret, sharperatio

class portfolio_optimizer:
    '''__________________________________________________________________________________________________________'''
    "________________________________________________Optimizer____________________________________________________"
    def __init__(self, ret, cov, risk_free = 1/100):
        self.ret = ret
        self.cov = cov
        self.cov_shape = cov.shape[0]
        #print (self.cov)
        cov_eig = np.linalg.eig(self.cov)[0]
        #print (cov_eig)
        #print (cov_eig > 0)
        if ((cov_eig >= 0).sum() == cov_eig.size).astype(np.int) == 1:
            print("The covariance matrix is at least PSD")
            pass
        else:
            print("The covariance matrix is not a PSD, and adjusted")
            smallest_eig = min(cov_eig)
            cov_new = self.cov - smallest_eig*np.identity(self.cov_shape)
            self.cov = cov_new/cov_new[0,0]
        self.risk_free = risk_free
        self.num_asset = self.cov_shape
    #checked
    def hessian_matrix_adjust(self, cov, method = "small_per"):
        #check if hession is psd
        cov_eig = np.linalg.eig(cov)[0]
        cov_shape = cov.shape[0]
        #print (cov_eig)
        if ((cov_eig >= 0).sum() == cov_eig.size).astype(np.int) == 1:
            print("The covariance matrix is at least PSD")
            cov_new = cov
        else:
            if method == "small_per":
                smallest_eig = min(cov_eig)
                cov_new = cov - smallest_eig*np.identity(cov_shape)
                cov_new = cov_new/cov_new[0,0]
            elif method == "zero_eig":
                w, v = np.linalg.eig(cov)
                w [w < 0] = 0
                d = np.diag(w)
                cov_new = np.dot(np.dot(v, d), v.T)
                cov_new = cov_new/cov_new[0,0]
            elif method == "rank_1_update":
                w, v = np.linalg.eig(cov)
                la = min(w)
                d = np.diag(w)
                Q = np.dot(v, v.T)
                cov_new = cov - la*Q
                cov_new = cov_new/cov_new[0,0]
        return cov_new
    #checked
    def min_variance(self, method = 'SLSQP'):

        #print (np.zeros((self.num_asset, 1)))
        local_cov = self.cov
        def obj_fun(x):
            obj = np.dot(np.dot(x.T, local_cov), x)
            return obj

        cons = ({'type': 'eq', 'fun': lambda x: np.ones((1, self.num_asset)).dot(x) - 1})
        bnd = [(0, None)]*self.num_asset
        res = spo.minimize(obj_fun, np.zeros((self.num_asset, 1)), method=method, bounds=bnd, constraints=cons, tol=1e-10)
        return res
    #checked
    def max_ret(self, method = 'SLSQP'):

        local_ret = self.ret
        #print (local_ret)
        def obj_fun(x):
            obj = -np.dot(local_ret.T, x)
            return obj

        cons = ({'type': 'eq', 'fun': lambda x: np.ones((1, self.num_asset)).dot(x) - 1})
        bnd = [(0, None)]*self.num_asset
        res = spo.minimize(obj_fun, np.zeros((self.num_asset, 1)), method=method, bounds=bnd, constraints=cons, tol=1e-10)
        return res
    #checked
    def mean_variance(self, method = 'SLSQP', plot_eff_front = False):
        min_var_x = self.min_variance().x.T
        min_var_ret = np.dot(self.ret.T, min_var_x)
        max_ret_x = self.max_ret().x.T
        max_ret_ret = np.dot(self.ret.T, max_ret_x)
        target_ret = np.linspace(min_var_ret, max_ret_ret, num=50)
        local_cov = self.cov
        local_ret = self.ret

        def obj_fun(x):
            obj = np.dot(np.dot(x.T, local_cov), x)
            return obj

        bnd = [(0, None)] * self.num_asset
        ret_list = []
        var_list = []
        res_list = []
        for item in target_ret:
            #print (item)
            cons = ({'type': 'eq', 'fun': lambda x: np.ones((1, self.num_asset)).dot(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: -item + np.dot(local_ret.T, x)})
            res = spo.minimize(obj_fun, np.zeros((self.num_asset, 1)), method=method, bounds=bnd, constraints=cons, tol=1e-10)
            res_list.append(res.x)
            ret_list.append(local_ret.T.dot(res.x.T))
            var_list.append(np.dot(np.dot(res.x, local_cov), res.x.T))

        if plot_eff_front == True:
            plt.figure()
            plt.plot(var_list[:-1], ret_list[:-1])
            plt.show()

        return ret_list, var_list, res_list
    #checked
    def max_sharpe(self, method = "SLSQP"):
        if (self.ret >= self.risk_free/252).sum() > 0:
            Q_temp_row = np.zeros((1, self.num_asset))
            Q_temp_col = np.zeros((self.num_asset + 1, 1))
            Q_new = np.hstack((np.vstack((self.cov, Q_temp_row)), Q_temp_col))
            eq_con_A1 = np.append((self.ret - self.risk_free/252), [0], axis=0)
            #print (eq_con_A1)
            eq_con_A2 = np.append(np.ones(self.num_asset), [-1], axis=0)
            #print (eq_con_A2)
            #print (-1*np.identity(self.num_asset))
            #ineq_con_A1 = np.hstack((-1*np.identity(self.num_asset), np.zeros((self.num_asset, 1))))
            #print (ineq_con_A1)
            #ineq_con_A2 = np.hstack((np.identity(self.num_asset), -1*np.ones((self.num_asset, 1))))
            #print (Q_new)

            def obj_fun(x):
                obj = np.dot(np.dot(x.T, Q_new), x)
                return obj

            cons = ({'type': 'eq', 'fun': lambda x: eq_con_A1.dot(x) - 1},
                    {'type': 'eq', 'fun': lambda x: eq_con_A2.dot(x)})

            #for item in range(self.num_asset):
                #print (ineq_con_A1[item, :])
                #print (ineq_con_A2[item, :])
                #tempcon1 = ({'type': 'ineq', 'fun': lambda x: x[item] - x[-1]},)
                #tempcon1 = ({'type': 'ineq', 'fun': lambda x: ineq_con_A1[item, :].dot(x)},)
                #cons = cons + tempcon1
                #tempcon2 = ({'type': 'ineq', 'fun': lambda x: ineq_con_A2[item, :].dot(x)},)
                #cons = cons + tempcon2
            #print (cons)
            bnd = [(0, None)] * (self.num_asset+1)
            res = spo.minimize(obj_fun, np.random.uniform(low=0.0, high=100, size=(self.num_asset+1, 1)),
                               method=method, bounds=bnd, constraints=cons, tol=1e-15)
            opt_weight = res.x[:-1]/res.x[-1]
            return opt_weight
        else:
            print ("All asset have lower returns than risk-free asset, max sharpe's ratio has no solution.")
            ret_list, var_list, res_list = self.mean_variance()
            sharperatio = (ret_list - self.risk_free/252) / np.sqrt(var_list)
            max_sharpe_index = np.argmax(sharperatio)
            max_sharpe = np.amax(sharperatio)
            opt_weight = res_list[max_sharpe_index]
            return opt_weight, max_sharpe_index
    # checked
    def equal_risk(self, method = "SLSQP"):
        local_Q = self.cov
        num_asset = self.num_asset
        def obj_fun(x):
            #print (np.dot(local_Q,x))
            term1 = x*local_Q.dot(x)
            term1 = term1.reshape(5,1)
            term1 = numpy.matlib.repmat(term1, 1, num_asset)
            term2 = x*local_Q.dot(x)
            term2 = term2.reshape(1,5)
            term2 = numpy.matlib.repmat(term2, num_asset, 1)
            f = np.sum(np.sum(np.power((term1 - term2), 2)))
            return f

        cons = ({'type': 'eq', 'fun': lambda x: np.ones((1, self.num_asset)).dot(x) - 1})
        bnd = [(0, None)]*self.num_asset
        res = spo.minimize(obj_fun, np.random.normal(loc=0.0, scale=1.0, size=(self.num_asset, 1)), method=method,
                           bounds=bnd, constraints=cons, tol=1e-15)
        risk = res.x*local_Q.dot(res.x)/np.sqrt(np.dot(np.dot(res.x, local_Q), res.x))
        return res.x, risk

    def equal_weight(self):
        res = (1/self.num_asset)*np.ones(self.num_asset)
        return res
    #checked
    def rob_mean_variance(self, method = "SLSQP"):
        local_Q = self.cov
        local_ret = self.ret
        num_asset = self.num_asset
        # initial weight
        w0 = np.ones((num_asset, 1))/num_asset
        #ret_init = np.dot(local_ret.T, w0)
        #var_init = np.dot(np.dot(w0.T, local_Q), w0)

        # target return estimation error
        var_matr = np.diag(np.diag(local_Q))
        #print (w0)
        rob_init = np.dot(np.dot(w0.T, var_matr), w0)
        rob_init = rob_init[0]
        port_ret = np.dot(local_ret.T, self.min_variance().x)
        #print (port_ret)

        def obj_fun(x):
            obj = np.dot(np.dot(x.T, local_Q), x)
            return obj

        cons = ({'type': 'eq', 'fun': lambda x: np.ones((1, self.num_asset)).dot(x) - 1},
                {'type': 'ineq', 'fun': lambda x: -port_ret + np.dot(local_ret.T, x)},
                {'type': 'ineq', 'fun': lambda x: np.dot(np.dot(x.T, var_matr), x) - rob_init})
        bnd = [(0, None)]*self.num_asset
        res = spo.minimize(obj_fun, np.random.uniform(low=0.0, high=1, size=(self.num_asset, 1)),
                           method=method, bounds=bnd, constraints=cons, tol=1e-15)
        return res

class price_predictor_adjuster:
    '''___________________________________________________________________________________________________________'''
    '''____________________________________________Machine Learning_________________________________________________'''
    def __init__(self, risk_free = 0.01):
        self.risk_free = risk_free

    def black_litterman(self, market_cap, historical_observation, P_sub, Q_sub, tau = 0.05, obs_period = 12):
        print ("The observation and prediction shall be at least in monthly basis")
        average_risk_free = self.risk_free/obs_period
        hist_mean = historical_observation[0] - average_risk_free
        hist_cov = historical_observation[1]
        num_asset = market_cap.shape[0]
        weight = market_cap/np.sum(market_cap)

        def BL_model(tau, P_sub, Q_sub):
            delta = weight.T.dot(hist_mean)/(np.dot(weight.T.dot(hist_cov), weight))
            mu_obj = delta*hist_cov.dot(weight)
            Q_obj = tau*hist_cov
            if ((P_sub == 0).sum() == P_sub.size).astype(np.int) == 1:
                mu_new = mu_obj
                cov_new = hist_cov + Q_obj
            else:
                omiga = np.dot(P_sub.dot(tau*hist_cov), P_sub.T)
                Omi = np.diag(np.diag(omiga))
                term1 = hist_cov.dot(P_sub.T)
                term2 = np.dot((P_sub*tau).dot(hist_cov), P_sub.T) + Omi
                term3 = Q_sub - P_sub.dot(mu_obj)
                mu_new = mu_obj + tau*np.dot(term1.dot(np.linalg.inv(term2)), term3)
                term4 = np.linalg.inv(tau*hist_cov) + np.dot(P_sub.T.dot(np.linalg.inv(Omi)), P_sub)
                cov_new = hist_cov + np.linalg.inv(term4)
            return mu_new, cov_new

        excessive_mu, cov_new = BL_model(tau, P_sub, Q_sub)
        mu_new = excessive_mu + average_risk_free

        return excessive_mu, mu_new, cov_new





'''
class option_price:


class trading_alg:


class risk_sim:'''


