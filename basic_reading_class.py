import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import urllib.request
from scipy import stats
import time
import os
import numpy.matlib

class finance_basic_info:
    """______________________________________________________________________________________________________________"""
    "______________________________________________Fetch Data and Basic Info___________________________________________"
    def __init__(self, company_list, risk_free= 1/100):
        self.company_list = company_list
        self.risk_free = risk_free
        if 'SPY' not in self.company_list:
            self.company_list.append('SPY')

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
    def get_data(self, start_date, end_date, benckmark = False, norm_plot = False):
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
        row_data = row_data.iloc[::-1]
        if benckmark is False:
            row_data = row_data.drop('SPY', axis=1)
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

    def tech_indicator(self, dataframe, n_day = 5, skip_nan = True):
        rolling_stats = self.rolling_stats(dataframe=dataframe, window=n_day)
        momentum = dataframe.copy()
        momentum [n_day:] = (dataframe[n_day:]/dataframe[:-n_day].values) - 1
        momentum.ix[0:n_day] = np.nan
        simple_moving_avg = dataframe.copy()
        simple_moving_avg [n_day-1:] = (dataframe[n_day-1:]/rolling_stats[0][n_day-1:].values) - 1
        simple_moving_avg[0:n_day-1] = np.nan
        #print(simple_moving_avg)
        bband = dataframe.copy()
        bband[n_day-1:] = (dataframe[n_day-1:].values - rolling_stats[0][n_day-1:].values)/(2*rolling_stats[1][n_day-1:].values)
        bband[0:n_day-1] = np.nan
        #print (bband)
        #normalized:
        momentum = (momentum - momentum.mean())/momentum.std()
        simple_moving_avg = (simple_moving_avg - simple_moving_avg.mean())/simple_moving_avg.std()
        bband = (bband - bband.mean())/bband.std()
        temp_1 = pd.Series(momentum, name='Momentum')
        temp_2 = pd.Series(simple_moving_avg, name='Simple Moving Average')
        temp_3 = pd.Series(bband, name='B Band')
        #print (temp_1.to_frame())
        #print (temp_2.to_frame())
        #print (temp_3.to_frame())
        res_table = temp_1.to_frame().join(temp_2.to_frame())
        res_table = res_table.join(temp_3.to_frame())
        if skip_nan == True:
            momentum = momentum.ix[n_day:]
            simple_moving_avg = simple_moving_avg.ix[n_day-1:]
            bband = bband.ix[n_day-1:]
            res_table = res_table.iloc[n_day:]
        else:
            pass
        return momentum, simple_moving_avg, bband, res_table

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
            term1 = term1.reshape(num_asset,1)
            term1 = numpy.matlib.repmat(term1, 1, num_asset)
            term2 = x*local_Q.dot(x)
            term2 = term2.reshape(1,num_asset)
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
    #checked
    def CAPM(self, market_ret, asset_ret, asset_weigth):
        num_asset = asset_weigth.shape[0]
        asset_alpha= []
        asset_beta = []
        try:
            asset_ret = asset_ret.to_frame()
        except:
            pass
        #print (asset_ret)
        for item in range(num_asset):
            #print (asset_ret.to_frame().ix[:, item])
            #print (asset_ret.ix[:, item].values)
            res = stats.linregress(market_ret.values, asset_ret.ix[:, item].values)
            beta = res[0]
            alpha = res[1]
            asset_alpha.append(alpha)
            asset_beta.append(beta)
        #print(asset_beta)
        #print(asset_alpha)
        port_beta = np.array(asset_beta).dot(asset_weigth)
        port_alpha = np.array(asset_alpha).dot(asset_weigth)
        return port_alpha, port_beta, asset_beta, asset_alpha
    #checked
    def KNN(self, raw_data_tech, raw_data_price, predict_data, k = None, day_predication = 5):
        raw_data_y = pd.Series(raw_data_price, name='Adj Close')
        raw_data_y = raw_data_y.to_frame()
        raw_data_x = raw_data_tech.copy()
        raw_data_y = raw_data_y.ix[day_predication:, ['Adj Close']]
        raw_data_x = raw_data_x.iloc[:-day_predication]
        #print (raw_data_x)
        num_data_x = raw_data_x.shape[0]
        num_data_y = raw_data_y.shape[0]
        if (num_data_x == num_data_y) is False:
            print ('The size of x and y is not same')
        else:
            pass
        if k is None:
            k = np.round(np.sqrt(num_data_x))
        raw_data_y.reset_index(inplace = True)
        tag = 'Price/RR in Next ' + str(day_predication) + ' Days'
        raw_data_y = raw_data_y.rename(columns={'Adj Close': tag})
        raw_data_y = raw_data_y.drop('index', axis=1)
        eulerian_dist = np.sum((raw_data_x - predict_data)**2, axis=1)
        eulerian = pd.Series(eulerian_dist, name='Eulerian Distance')
        eulerian = eulerian.to_frame()
        eulerian.reset_index(inplace=True)
        combined_new_data = eulerian.join(raw_data_y)
        combined_new_data.sort_values(by='Eulerian Distance', inplace=True)
        combined_new_data.reset_index(inplace = True)
        combined_new_data = combined_new_data.drop('level_0', axis=1)
        k_predict = combined_new_data.ix[:k, [tag]]
        res_mean = k_predict.mean()
        res_std = k_predict.std()
        print (res_mean)
        return res_mean, res_std
'''
    def Q_learning(self, train_data, num_step):
        step_size = np.round(train_data.shape[0]/num_step)
        num_feature = train_data.shape[1]

        def factor_discretize(factor):
            factor.sort()
            index_factor = factor.copy()
            threshold = np.array(1, num_step)
            for i in range(num_step):
                threshold[i] = factor[(i + 1) * step_size]
                if i == 0:
                    index_factor[index_factor <= threshold[i]] = i
                else:
                    index_factor[index_factor <= threshold[i] and index_factor > threshold[i-1]] = i
            return index_factor

        while True:

     def FCNN(self, data, activation_function = 'tanh'):'''

'''
    def back_test(self):
        
    def confident_interval(self):
        


class option_price:


class trading_alg:


class risk_sim:'''


