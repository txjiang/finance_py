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
        return  dailyret, meanret, stdret, covret, kurtret, sharperatio
    '''__________________________________________________________________________________________________________'''
    "________________________________________________Optimizer____________________________________________________"



    '''___________________________________________________________________________________________________________'''
    '''____________________________________________Machine Learning_________________________________________________'''


