import pandas as pd

run_date = '0421'
M = 30
gen = 10
te = pd.read_csv('/root/thesis/output/backtest_tracking_error_{0}_M{1}_gen{2}.csv'.format(run_date,M,gen), index_col=0)
# te = pd.read_csv('/root/thesis/output0/backtest_tracking_error_simple_GA_M30_gen10.csv', index_col=0)

te = te.set_index('date')

print('mean:\n')
print(te.mean(axis=0),'\n')
print('std:')
print(te.std(axis=0),'\n')