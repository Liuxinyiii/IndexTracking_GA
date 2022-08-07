import json
import pandas as pd
import tushare as ts
from tqdm import tqdm
from jqdatasdk import auth, get_index_weights, get_query_count, finance, get_industries, query
auth('***','***')
#
# # 计算行业因子
# with open('/root/thesis/data/all_stocks_industry.json', encoding='utf-8') as f:
#     industry_dict = json.load(f)
#     f.close()
# industry_data = pd.DataFrame()
# for stk in list(industry_dict.keys()):
#     try:
#         industry_data = industry_data.append(pd.DataFrame({'Stock': stk.split('.')[0],
#                                                            'Industry': industry_dict[stk]['sw_l1']['industry_code']},
#                                                           index=[0]))
#     except KeyError:
#         print(stk, '无对应申万一级行业')
#
# idx_compo = pd.read_csv('/root/thesis/data/index_compo_data.csv')
# idx_compo['开始日期_BegDt'] = pd.to_datetime(idx_compo['开始日期_BegDt'].apply(str))
# idx_compo['结束日期_EndDt'] = pd.to_datetime(idx_compo['结束日期_EndDt'].apply(str))
# index_factors = pd.read_csv('/root/thesis/data/index_factors.csv', index_col=0)
# index_factors_new = index_factors.set_index('date')
#
# pro = ts.pro_api('432455924591f5b8bc3a633340ccb0e719d7ab1edf1b939542329b50')
#
# for industry in list(set(list(industry_data['Industry']))):
#     index_factors[industry] = 0
#
# for date in tqdm(index_factors.date):
#     date_dt = pd.to_datetime(date)
#     compo_weight = get_index_weights(index_id='399300.XSHE', date=date)
#     compo_weight = compo_weight.reset_index().rename(columns={'index': 'Stock'})
#     compo_weight.Stock = compo_weight.Stock.apply(lambda x: x.split('.')[0])
#     compo_weight = pd.merge(compo_weight, industry_data, on='Stock')
#     for industry in list(set(list(industry_data['Industry']))):
#         try:
#             ind_weight = compo_weight.groupby('Industry').sum().loc[industry, 'weight']
#         except KeyError:
#             continue
#         index_factors_new.loc[date, industry] = ind_weight

# 获取行业指数作为因子
pro = ts.pro_api('432455924591f5b8bc3a633340ccb0e719d7ab1edf1b939542329b50')
index_factors = pd.read_csv('/root/thesis/data/index_factors.csv', index_col=0)
sw_names = ['801180','801790','801150','801120','801750','801160','801040','801030','801020','801780']
for sw_ind in tqdm(sw_names):
    sw_ind_price = finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == sw_ind).order_by(
        finance.SW1_DAILY_PRICE.date.desc()).limit(2740))[['close','date']]
    sw_ind_price.date = sw_ind_price.date.apply(lambda x: x.strftime('%Y-%m-%d'))
    sw_ind_price['pct'] = sw_ind_price['close'].pct_change(-1) * 100
    sw_ind_price = sw_ind_price.rename(columns={'pct': sw_ind})
    index_factors = pd.merge(index_factors,sw_ind_price[[sw_ind,'date']], on='date',how='outer')
index_factors.to_csv('/root/thesis/data/index_factors_with_industry_pct.csv')

