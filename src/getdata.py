import pandas as pd
import numpy as np
import tushare as ts
from jqdatasdk import get_factor_values, auth
import time
from tqdm import tqdm

# %%
startdate = '20160101'
enddate = '20210101'
pro = ts.pro_api('***')

# %%
# 指数日度数据
# df = pro.index_daily(ts_code='000300.SH', start_date='20100101', end_date='20210101')
#
# df.to_csv('/Users/liuxinyi/Desktop/论文/data/HS300.csv',encoding='utf-8')

# %%
# # 指数成分股数据
# pro = ts.pro_api('432455924591f5b8bc3a633340ccb0e719d7ab1edf1b939542329b50')
# result = pd.DataFrame()
# for year in [x for x in range(10,22)]:
#     for month in [x for x in range(1,13)]:
#         df = pro.index_weight(index_code='399300.SZ', start_date='20{0}{1}01'.format(year,month), end_date='20{0}{1}30'.format(year,month))

# %%
# # 成分股日度数据
# pro = ts.pro_api('432455924591f5b8bc3a633340ccb0e719d7ab1edf1b939542329b50')
# idx_compo = pd.read_csv('/Users/liuxinyi/Desktop/论文/data/index_compo_data.csv')
# for i in tqdm(range(383,len(idx_compo))):
#     code = str(idx_compo['成分股代码_CompoStkCd'].iloc[i]).zfill(6)
#     df = pro.daily(ts_code=code + '.SZ', start_date='20091201', end_date='20210101')
#     if len(df) > 1:
#         df.to_csv('/Users/liuxinyi/Desktop/论文/data/stk_data/{0}.csv'.format(code))
#     else:
#         df = pro.daily(ts_code=code + '.SH', start_date='20091201', end_date='20210101')
#         df.to_csv('/Users/liuxinyi/Desktop/论文/data/stk_data/{0}.csv'.format(code))
#     time.sleep(1)

# %% 因子数据
# auth('15201141906', 'Zhengrenlianghua2020')
pro = ts.pro_api('432455924591f5b8bc3a633340ccb0e719d7ab1edf1b939542329b50')

ff3 = pd.read_csv('/Users/liuxinyi/Desktop/论文/data/ff3_0.csv',index_col=0).append(pd.read_csv('/Users/liuxinyi/Desktop/论文/data/ff3_1.csv',index_col=0))
ff3.date = pd.to_datetime(ff3.date.apply(str))
barra_factors_all = pd.read_csv('/Users/liuxinyi/Desktop/正仁量化/zr/data/barra/barra_factors_20160101_to_20210101.csv', index_col=0)
all_dates = list(set(list(barra_factors_all.Date)))
all_dates.sort()
index_barra_factors = pd.DataFrame()
for date in tqdm(all_dates[202:]):
    df = pro.index_weight(index_code='399300.SZ', end_date=date)
    df.trade_date = pd.to_datetime(df.trade_date)
    extract_date = list(set(list(df.trade_date)))
    extract_date.sort()
    nearest_date = extract_date[-1]
    weight = df[df.trade_date == nearest_date]
    day_barra_factors = barra_factors_all[barra_factors_all.Date == date]
    day_barra_factors['weight'] = 0
    day_barra_factors = day_barra_factors.set_index('Stock')
    for row in weight.iterrows():
        day_barra_factors.loc[int(row[1].con_code.split('.')[0]),'weight'] = row[1].weight
    del day_barra_factors['Date']
    day_result = day_barra_factors.mul(day_barra_factors['weight'],axis=0).sum()
    day_result['date'] = date
    index_barra_factors = index_barra_factors.append(day_result, ignore_index=True)
    time.sleep(5)
