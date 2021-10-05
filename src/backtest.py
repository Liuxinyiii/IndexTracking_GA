import sys
sys.path.append('/root/thesis')
import json
import time
import pandas as pd
import numpy as np
import tushare as ts
from tqdm import tqdm
from Code.src.model import GeneticAlgorithm

# %% 参数设定
run_date = 'AGA_with_industry'
print(run_date)
M = 30
gen = 20
pool_num = 300
period = 20
start_date = '20170101'
end_date = '20200101'

pro = ts.pro_api('432455924591f5b8bc3a633340ccb0e719d7ab1edf1b939542329b50')
trade_cal = pro.trade_cal(exchange='', start_date=start_date , end_date=end_date)
all_trade_dates = trade_cal[trade_cal.is_open == 1].reset_index(drop=True)
month_loc = [period * x for x in range(len(all_trade_dates)//period)]
test_dates = list(all_trade_dates.iloc[month_loc,:].cal_date)

# %% GA run
print('开始运行遗传算法...')

time_start = time.time()
holding = pd.DataFrame(columns=[x for x in range(0, 300)])  # 注意pool变动，所对应的成分股需要重新找 idx_compo[idx_compo['开始日期_BegDt'] < date_dt][idx_compo['结束日期_EndDt'] != idx_compo['结束日期_EndDt']].append(idx_compo[idx_compo['开始日期_BegDt'] < date_dt][idx_compo['结束日期_EndDt'] == idx_compo['结束日期_EndDt']][idx_compo['结束日期_EndDt'] >= date_dt])
holding.to_csv('/root/thesis/output/backtest_holding_{0}_M{1}_gen{2}.csv'.format(run_date,M,gen), encoding='utf-8')
record = pd.DataFrame(columns=['winner_f','runtime'])
record.to_csv('/root/thesis/output/backtest_record_{0}_M{1}_gen{2}.csv'.format(run_date,M,gen), encoding='utf-8')

for date in test_dates:
    print('testing:', date)
    time_start1 = time.time()
    ga = GeneticAlgorithm(M=M, gen=gen, pool_num=pool_num, date=date, use_AGA=True)
    ga.run()
    f = [ga.fitness(ga.X[i]) for i in range(ga.M)]
    fitness_sort = sorted(enumerate(f), key=lambda x: x[1], reverse=True)
    rank = [i[0] for i in fitness_sort]
    winner_f = f[rank[0]]
    winner = ga.X[rank[0]]
    time_end1 = time.time()
    ind_holding = pd.DataFrame(data=np.array(winner).reshape(1, 300), index=[date])
    ind_holding.to_csv('/root/thesis/output/backtest_holding_{0}_M{1}_gen{2}.csv'.format(run_date,M,gen), encoding='utf-8',mode='a',header=False)
    holding = holding.append(ind_holding)
    ind_record = pd.DataFrame({'winner_f': winner_f, 'runtime': round((time_end1 - time_start1) / 3600, 2)}, index=[date])
    ind_record.to_csv('/root/thesis/output/backtest_record_{0}_M{1}_gen{2}.csv'.format(run_date,M,gen), encoding='utf-8',mode='a',header=False)
    record = record.append(ind_record)

# holding.to_csv('/root/thesis/output/backtest_holding_0311.csv', encoding='utf-8')
# record.to_csv('/root/thesis/output/backtest_record_0311.csv', encoding='utf-8')

time_end = time.time()
print('优化完成！Time spent:',round((time_end - time_start) / 3600, 2), 'hours')


# %% Backtest

print('开始回测....')
time_start = time.time()

idx_compo = pd.read_csv('/root/thesis/data/index_compo_data.csv')
idx_price = pd.read_csv('/root/thesis/data/HS300.csv',index_col=0)
idx_compo['开始日期_BegDt'] = pd.to_datetime(idx_compo['开始日期_BegDt'].apply(str))
idx_compo['结束日期_EndDt'] = pd.to_datetime(idx_compo['结束日期_EndDt'].apply(str))
idx_price.trade_date = pd.to_datetime(idx_price.trade_date.apply(str))
all_trade_dates.cal_date = pd.to_datetime(all_trade_dates.cal_date)

with open('/root/thesis/data/all_stocks_industry.json', encoding='utf-8') as f:
    industry_dict = json.load(f)
    f.close()
industry_data = pd.DataFrame()
for stk in list(industry_dict.keys()):
    try:
        industry_data = industry_data.append(pd.DataFrame({'Stock': stk.split('.')[0],
                                                           'Industry': industry_dict[stk]['sw_l1']['industry_code']},
                                                          index=[0]))
    except KeyError:
        print(stk, '无对应申万一级行业')


holding_record = '/root/thesis/output/backtest_holding_{0}_M{1}_gen{2}.csv'.format(run_date, M, gen)
holding = pd.read_csv(holding_record, encoding='utf-8', index_col=0)

return_all = pd.DataFrame()
te_all = pd.DataFrame()
for i in tqdm(range(len(list(holding.index)))):
    date = list(holding.index)[i]
    date_dt = pd.to_datetime(str(date))
    date_cal_loc = all_trade_dates.query('cal_date==@date_dt').index.values[0]
    period_dates = all_trade_dates.iloc[date_cal_loc + 1:date_cal_loc + period+1, :]
    compo_stk = idx_compo[idx_compo['开始日期_BegDt'] < date_dt][idx_compo['结束日期_EndDt'] != idx_compo['结束日期_EndDt']].append(idx_compo[idx_compo['开始日期_BegDt'] < date_dt][idx_compo['结束日期_EndDt'] == idx_compo['结束日期_EndDt']][idx_compo['结束日期_EndDt'] >= date_dt])  # 遗传算法计算时未用到当天数据
    compo_stk = compo_stk.sort_index()

    # 传统市值法 holding
    size = compo_stk.copy()
    size['mv'] = None
    for j in range(len(size)):
        code = str(size.iloc[j,0]).zfill(6)
        mv_data = pro.daily_basic(ts_code=code+'.SH', trade_date=str(date), fields='ts_code,trade_date,total_mv')
        if len(mv_data) > 0:
            size.iloc[j,3] = mv_data.total_mv.values[0]
        else:
            mv_data = pro.daily_basic(ts_code=code+'.SZ', trade_date=str(date), fields='ts_code,trade_date,total_mv')
            if len(mv_data) > 0:
                size.iloc[j,3] = mv_data.total_mv.values[0]
        time.sleep(0.5)
    size = size.sort_values(by='mv', ascending=False)
    size['scaled_weight'] = None
    for m in range(len(size)):
        if m < 60:
            size.iloc[m, 4] = size.iloc[m, 3] / size.iloc[:60, 3].sum()
        else:
            size.iloc[m, 4] = 0
    size_day_holding = []
    for k in range(len(compo_stk)):
        Cd = compo_stk.iloc[k,0]
        size_day_holding.append(size.query("成分股代码_CompoStkCd == @Cd")['scaled_weight'].values[0])

    # 传统行业分层法 holding
    stratified = size.copy()
    stratified['scaled_weight'] = 0
    stratified['成分股代码_CompoStkCd'] = stratified['成分股代码_CompoStkCd'].apply(lambda x: str(x).zfill(6))
    stratified = stratified.rename(columns={'成分股代码_CompoStkCd': 'Stock'})
    stratified = pd.merge(stratified, industry_data, on='Stock')
    industry_portion = stratified.fillna(0).groupby('Industry').sum()['mv']
    industry_portion = industry_portion * (60 / 300)
    stratified = stratified.set_index('Stock')
    for industry in industry_portion.index:
        ind_data = stratified[stratified['Industry'] == industry]
        ind_data = ind_data.sort_values(by='mv',ascending=False)
        ind_data['mv2'] = ind_data['mv'].cumsum()
        ind_data['mv2'] = ind_data['mv2'] / industry_portion.loc[industry]
        ind_num = (industry_portion / industry_portion.sum() * 60).apply(lambda x: round(x)).loc[industry]
        ind_data = ind_data.iloc[:ind_num,:]
        ind_data['scaled_weight'] = ind_data['mv'] / ind_data['mv'].sum()  # 行业内部weight
        ind_data['scaled_weight'] = ind_data['scaled_weight'] * (industry_portion / industry_portion.sum()).loc[industry]
        for ind_data_stk in ind_data.index:
            stratified.loc[ind_data_stk,'scaled_weight'] = ind_data.loc[ind_data_stk,'scaled_weight']
    stratified_day_holding = []
    for k in range(len(compo_stk)):
        Cd = compo_stk.iloc[k,0]
        try:
            stratified_day_holding.append(stratified.loc[str(Cd).zfill(6), 'scaled_weight'])
        except KeyError:
            stratified_day_holding.append(0)

    ga_day_holding = list(holding.loc[date, :])
    ga_port_ret = np.array([float(0) for x in range(period)])
    size_port_ret = np.array([float(0) for x in range(period)])
    stratified_port_ret = np.array([float(0) for x in range(period)])

    for x in range(pool_num):
        code = str(compo_stk['成分股代码_CompoStkCd'].iloc[x]).zfill(6)
        df = pd.read_csv('/root/thesis/data/stk_data/{0}.csv'.format(code), index_col=0)
        df.trade_date = pd.to_datetime(df.trade_date.apply(str))
        df = df[df.trade_date > date_dt]
        stk_ret = np.array(list(df.iloc[-period:, :]['pct_chg'].values) + [0 for x in range(period - len(df.iloc[-period:, :]['pct_chg'].values))]).reshape(20,)
        ga_port_ret += (ga_day_holding[x] / sum(ga_day_holding)) * stk_ret
        size_port_ret += (size_day_holding[x] / sum(size_day_holding)) * stk_ret
        stratified_port_ret += (stratified_day_holding[x] / sum(stratified_day_holding)) * stk_ret

    idx_ret = idx_price.query("trade_date > @date_dt").iloc[-period:,:]['pct_chg'].values  # 时间倒序，第一个值为最新
    ga_te = (sum((idx_ret - ga_port_ret - (idx_ret - ga_port_ret).mean()) ** 2) / period) ** 0.5
    size_te = (sum((idx_ret - size_port_ret - (idx_ret - size_port_ret).mean()) ** 2) / period) ** 0.5
    stratified_te = (sum((idx_ret - stratified_port_ret - (idx_ret - stratified_port_ret).mean()) ** 2) / period) ** 0.5
    period_result_df = pd.DataFrame(index=period_dates.cal_date,columns=['GAReturn','SizeReturn','StratifiedReturn','IndexReturn'])
    period_result_df['GAReturn'] = ga_port_ret[::-1]
    period_result_df['IndexReturn'] = idx_ret[::-1]
    period_result_df['SizeReturn'] = size_port_ret[::-1]
    period_result_df['StratifiedReturn'] = stratified_port_ret[::-1]

    return_all = return_all.append(period_result_df)
    te_all = te_all.append(pd.DataFrame({'date': date_dt, 'ga_te': ga_te, 'size_te': size_te, 'stratified_te':stratified_te}, index=[0]))

return_all.to_csv('/root/thesis/output/backtest_return_{0}_M{1}_gen{2}.csv'.format(run_date,M,gen),encoding='utf-8')
te_all.to_csv('/root/thesis/output/backtest_tracking_error_{0}_M{1}_gen{2}.csv'.format(run_date,M,gen),encoding='utf-8')

time_end = time.time()
print('回测完成！Time spent:',round((time_end - time_start) / 3600, 2), 'hours')
