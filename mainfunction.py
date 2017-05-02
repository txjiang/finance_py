from basic_reading_class import finance_basic_info
from basic_reading_class import portfolio_optimizer
from basic_reading_class import price_predictor_adjuster
import numpy as np

# Test FBI
company_list = ['AMD', 'BAC', 'GOOG', 'MSFT', 'TXN']
fbi = finance_basic_info(company_list=company_list)
#fbi.pullData()
start_date = '2016-05-02'
end_date = '2017-01-09'
rowdata = fbi.get_data(start_date = start_date, end_date = end_date, benckmark= True, norm_plot=False)
#print (rowdata)
daily_ret, mean_ret, std, cov, kurtret, sharperatio = fbi.dailyret_stats(rowdata)
#print (daily_ret)
AMD = daily_ret['AMD']
SPY = daily_ret['SPY']
BAC = daily_ret['BAC']
portf = AMD.to_frame().join(BAC)
#print(portf)
#print (AMD)
#print (SPY)
#t1, _, t, tech = fbi.tech_indicator(AMD, skip_nan=True)
#AMD = AMD.loc['2016-05-09':]
#print (AMD)
#print(tech)
#print (AMD)
#print (daily_ret)
#rowdata = rowdata.iloc[::-1]
#print(rowdata)
#res = fbi.tech_indicator(rowdata)
#print (res)
# Test po
#cov_test = np.array([[1, 0.3, 0.5], [0.3, 1, -0.8], [0.5, -0.8, 1]])
#ret_test = np.array([0.01, -0.002, 0.009])
#po = portfolio_optimizer(ret = mean_ret, cov=cov)
#x = po.hessian_matrix_adjust(cov = cov)
#res = po.min_variance(method='SLSQP')
#res = po.mean_variance(plot_eff_front=True)
#res = po.max_ret()
#res = po.max_sharpe()
#res = po.equal_risk()
#res = po.rob_mean_variance()
#print (res)
#print(x)
#pred_data = tech.iloc[-1]
#print (pred_data)
ppa = price_predictor_adjuster()
w = np.array([0.5, 0.5])
res = ppa.CAPM(market_ret=SPY, asset_ret=portf, asset_weigth=w)
#res = ppa.KNN(raw_data_tech=tech, raw_data_price=AMD, predict_data=pred_data, day_predication=2)
print(res)
