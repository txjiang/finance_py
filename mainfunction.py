from basic_reading_class import finance_basic_info

company_list = ['AMD', 'BAC', 'GOOG', 'MSFT', 'TXN']
fbi = finance_basic_info(company_list=company_list)
#fbi.pullData(length = '1y')
start_date = '2016-10-02'
end_date = '2017-01-09'
rowdata = fbi.get_data(start_date = start_date, end_date = end_date)
#print (rowdata)
rm, _, _, _ = fbi.rolling_stats(rowdata)
dr, _, std, cov, _, sharperatio = fbi.dailyret_stats(rowdata)
print(cov)

