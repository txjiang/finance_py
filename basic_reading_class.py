import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import urllib.request
import time
import os
import csv

class finance_basic_info:
    """______________________________________________________________________________________________________________"""
    "______________________________________________Fetch Data and Basic Info___________________________________________"
    def __init__(self, company_list, risk_free= 1/100):
        self.company_list = company_list
        self.risk_free = risk_free

    # checked
    def pullData(self, length='2y'):
        for stock in self.company_list:
            fileLine = stock + '.csv'
            urltovisit = 'http://chartapi.finance.yahoo.com/instrument/1.0/' + stock + '/chartdata;type=quote;range=' + length +'/csv'
            with urllib.request.urlopen(urltovisit) as f:
                sourceCode = f.read().decode('utf-8')
            splitSource = sourceCode.split('\n')

            try:
                os.remove(fileLine)
            except OSError:
                pass

            for eachLine in splitSource:
                splitLine = eachLine.split(',')  # <---(here ',' instead of '.')
                if len(splitLine) == 6:  # <----( here, 6 instead of 5 )
                    if 'values' not in eachLine:
                        saveFile = open(fileLine, 'a')
                        linetoWrite = eachLine + '\n'
                        saveFile.write(linetoWrite)

            with open(stock + '.csv', newline='') as f:
                r = csv.reader(f)
                data = [line for line in r]
            with open(stock + '.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
                w.writerows(data)

            print('Pulled', stock)
            print('...')
            time.sleep(.5)

    # checked
    def read_data(self, filename):

        yahoo_finance = pd.read_csv(filename, index_col="Date",
                                          parse_dates=True, usecols=['Date', 'Close'],
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
            temp_dataframe = temp_dataframe.rename(columns={'Close': item[:-4]})
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

'''    def max_sharpe(self):

    def robust_mean_variance(self):

    def equal_risk(self):

    def buy_hold(self):

    def equal_weight(self):


class price_predictor_adjuster: ###black-litterman
    #'''#___________________________________________________________________________________________________________'''
    #____________________________________________Machine Learning_________________________________________________#
'''

    def __init__(self):

    def black_litterman(self):


class option_price:


class trading_alg:


class risk_sim:'''


