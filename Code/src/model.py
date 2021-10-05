import math
import numba
import time
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd
import numpy as np
import statsmodels.api as sm
from random import random, randint, uniform

import warnings
warnings.filterwarnings('ignore')

# %%
idx_compo = pd.read_csv('/root/thesis/data/index_compo_data.csv')
idx_price = pd.read_csv('/root/thesis/data/HS300.csv',index_col=0)
idx_compo['开始日期_BegDt'] = pd.to_datetime(idx_compo['开始日期_BegDt'].apply(str))
idx_compo['结束日期_EndDt'] = pd.to_datetime(idx_compo['结束日期_EndDt'].apply(str))
idx_price.trade_date = pd.to_datetime(idx_price.trade_date.apply(str))
factors = pd.read_csv('/root/thesis/data/index_factors_with_industry_pct.csv', index_col=0)
factors = factors.fillna(0)
factors.date = pd.to_datetime(factors.date.apply(str))
pro = ts.pro_api('432455924591f5b8bc3a633340ccb0e719d7ab1edf1b939542329b50')

# %%
# 成分股和目标指数beta计算（换仓滚动更新beta）
def cal_index_beta(date):
    date_dt = pd.to_datetime(date)
    idx_ret = idx_price[idx_price.trade_date < date_dt].iloc[:40]['pct_chg'].values
    factor = factors[factors.date < date_dt].iloc[-40:, :]
    factor_names = list(factor.columns)[1:]
    X = sm.add_constant(factor[factor_names].values)
    y = idx_ret.reshape(-1,1)
    model = sm.OLS(y, X)
    results = model.fit()
    beta = []
    for i in range(1,len(results.params)):
        beta.append(results.params[i])
    return beta

def cal_beta(weight,date):
    date_dt = pd.to_datetime(date)
    compo_stk = idx_compo[idx_compo['开始日期_BegDt'] < date_dt][idx_compo['结束日期_EndDt'] != idx_compo['结束日期_EndDt']].append(idx_compo[idx_compo['开始日期_BegDt'] < date_dt][idx_compo['结束日期_EndDt'] == idx_compo['结束日期_EndDt']][idx_compo['结束日期_EndDt'] >= date_dt])
    if len(compo_stk) != 300:
        print('未找到正确对应的指数成分股')
        raise ValueError

    beta = np.zeros(len(list(factors.columns)[1:]))
    for i in range(300):
        if weight[i] == 0:
            continue
        code = str(compo_stk['成分股代码_CompoStkCd'].iloc[i]).zfill(6)
        df = pd.read_csv('/root/thesis/data/stk_data/{0}.csv'.format(code),index_col=0)
        df.trade_date = pd.to_datetime(df.trade_date.apply(str))
        df = df[df.trade_date < date_dt]
        stk_ret = df.iloc[:40]['pct_chg'].values
        factor = factors[factors.date < date_dt].iloc[-40:,:]
        factor_names = list(factor.columns)[1:]
        X = sm.add_constant(factor[factor_names].values)
        y = stk_ret.reshape(-1,1)
        y = y[::-1]
        model = sm.OLS(y, X)
        results = model.fit()
        beta0 = []
        for x in range(1, len(results.params)):
            beta0.append(results.params[x])
        beta += (weight[i] / sum(weight)) * np.array(beta0)

    return list(beta)

# %% GA
class GeneticAlgorithm:
    def __init__(self, M, gen, pool_num, date, Pc=0.85, Pm=0.005, use_AGA=False):
        self.M = M  # 种群数量
        self.dec_num = 6  # 每个基因的位数，即每支股票购买的手数范围为0～99999手
        self.pool_num = 300
        self.Pc = Pc  # 交叉概率
        self.Pm = Pm  # 变异概率
        self.date = date  # 运行日期
        self.beta_thres = 0.1   # beta控制阈值
        self.idx_beta = cal_index_beta(self.date)
        self.pool_num = pool_num  # 股票池股票数量
        self.gen = gen  # 迭代次数
        self.X = []  # 种群
        self.X_chr = []  # 编码后种群
        self.f = []  # 种群适应度序列
        self.rank = []  # 种群适应度排名

        self.use_AGA = use_AGA  # 是否使用自适应遗传算法
        self.history = {  # 历史迭代最优个体记录
            'f': [],
            'x': [],
            'y': []
        }

    # 适应度函数 tracking error 样本为最近40个交易日日度数据
    def fitness(self,weight):
        date = self.date
        date_dt = pd.to_datetime(date)
        idx_ret = idx_price.query("trade_date < @date_dt").iloc[:40, :]['pct_chg'].values  # 时间倒序，第一个值为最新
        compo_stk = idx_compo[idx_compo['开始日期_BegDt'] < date_dt][
            idx_compo['结束日期_EndDt'] != idx_compo['结束日期_EndDt']].append(
            idx_compo[idx_compo['开始日期_BegDt'] < date_dt][idx_compo['结束日期_EndDt'] == idx_compo['结束日期_EndDt']][
                idx_compo['结束日期_EndDt'] >= date_dt])
        compo_stk = compo_stk.sort_index()
        if len(compo_stk) != 300:
            print('未找到正确对应的指数成分股')
            raise ValueError

        port_ret = np.array([0 for x in range(40)])
        for i in range(300):
            if weight[i] == 0:
                continue
            code = str(compo_stk['成分股代码_CompoStkCd'].iloc[i]).zfill(6)
            df = pd.read_csv('/root/thesis/data/stk_data/{0}.csv'.format(code), index_col=0)
            df.trade_date = pd.to_datetime(df.trade_date.apply(str))
            df = df[df.trade_date < date_dt]
            stk_ret = df.iloc[:40]['pct_chg'].values
            port_ret = port_ret + (weight[i] / sum(weight)) * stk_ret

        f = (sum((idx_ret - port_ret - (idx_ret - port_ret).mean()) ** 2) / 39) ** 0.5

        # 将beta条件加入罚函数 欧式距离
        beta = cal_beta(weight, date)
        idx_beta = self.idx_beta
        diff = 0
        for i in range(len(beta)):
            # diff += abs((beta[i] - idx_beta[i]) / idx_beta[i])
            diff += ((beta[i] - idx_beta[i]) / idx_beta[i]) ** 2
        # diff = 0
        return 1 / (f + (diff ** (1/2)))  # 适应度需满足越大越优

    # 自定义编码方式
    # 第1...n1是第一个值，2...n2是第二个值，采用整数手的数值
    def num2str(self, num):
        return str(num).zfill(self.dec_num)

    def encoder(self,x):
        chr_list = []
        for i in range(len(x)):
            individual = ''
            for j in x[i]:
                individual = individual + self.num2str(j)
            chr_list.append(individual)
        return chr_list

    def str2num(self,s):
        return int(s)

    def decoder(self,x_chr):
        x = []
        for ind_chr in x_chr:
            individual = []
            for i in range(self.pool_num):
                individual.append(self.str2num(ind_chr[i*self.dec_num:((i+1)*self.dec_num)]))
            x.append(individual)
        return x

    def if_meet_constraints(self,x):

        # 是否满足beta条件 注意此处beta条件限制为百分比（去除因子本身量纲影响）
        # 选择的股票个数是否小于300
        # if x.count(0) > 0:
        #     beta = cal_beta(x,self.date)
        #     idx_beta = self.idx_beta
        #     for i in range(len(beta)):
        #         diff = abs((beta[i]-idx_beta[i])/idx_beta[i])
        #         print(diff)
        #         if diff < self.beta_thres:
        #             continue
        #         else:
        #             return False
        #     return True
        # else:
        #     return False

        if x.count(0) > 0:
            return True
        else:
            return False

    # 选择
    def choose(self):
        # calculate percentage
        s = sum(self.f)
        p = [self.f[i] / s for i in range(self.M)]
        chosen = []
        # choose M times
        for i in range(self.M):
            cum = 0
            m = random()
            # Roulette
            for j in range(self.M):
                cum += p[j]
                if cum >= m:
                    if self.if_meet_constraints(self.X[j]):
                        chosen.append(self.X_chr[j])
                        break
        return chosen

    # TODO AGA

    # 交叉
    def crossover(self, chr):
        crossed = []
        # if chr list is odd
        # if len(chr) % 2:
        #     crossed.append(chr.pop())
        for i in range(0, len(chr)):#, 2):
            a = chr[i]
            try:
                b = chr[i + 1]
            except IndexError:
                b = chr[0]
            # 0.85 probability of crossover
            if self.use_AGA:
                if self.f[i] < np.mean(self.f):
                    Pc = 1
                else:
                    Pc = (np.max(self.f) - self.f[i]) / (np.max(self.f) - np.min(self.f))
            else:
                Pc = self.Pc

            if random() < Pc:
                loc = randint(1, len(chr[i]) - 1)
                # temp = a[loc:]
                a = a[:loc] + b[loc:]
                # b = b[:loc] + temp
            # add to crossed
            crossed.append(a)
            # crossed.append(b)
        return crossed

    # 变异
    def mutation(self, chr):
        res = []
        for i in range(len(chr)):
            l = list(chr[i])
            if self.use_AGA:
                if self.f[i] < np.mean(self.f):
                    Pm = 0.5
                else:
                    Pm = 0.5 * ((np.max(self.f) - self.f[i]) / (np.max(self.f) - np.min(self.f)))
            else:
                Pm = self.Pm

            for j in range(len(l)):
                # 0.05 probability of mutation on each location
                if random() < Pm:
                    while True:
                        r = str(randint(0, 9))
                        if r != l[j]:
                            l[j] = r
                            break
            res.append(''.join(l))
        return res

    def run(self):
        # 初始化种群
        x = []
        count = 0
        while True:
            individual = []
            for j in range(self.pool_num):
                if (randint(0,self.pool_num) + 3) < randint(0,self.pool_num):  # 随机不买入某些股票，使得概率略小于0.5
                    individual.append(0)
                else:
                    individual.append(randint(0,10**(self.dec_num-1)-1))
            if self.if_meet_constraints(individual):  # 初始种群满足条件
                x.append(individual)
                count += 1
                #print('init population:',count,'of',self.M)
            if count >= self.M:
                break
        self.X = x
        self.X_chr = self.encoder(x)

        # 迭代
        for iter in range(self.gen):
            # 计算适应度并排序
            self.f = [self.fitness(self.X[i]) for i in range(self.M)]
            fitness_sort = sorted(enumerate(self.f), key=lambda x: x[1], reverse=True)
            self.rank = [i[0] for i in fitness_sort]
            winner = self.f[self.rank[0]]
            print(f'Iter={iter + 1}, Max-Fitness={winner}')

            # 记录最优个体
            self.history['f'].append(winner)
            self.history['x'].append(self.X[self.rank[0]])

            # 交叉、变异、选择
            chosen = self.choose()
            crossed = self.crossover(chosen)
            new_X_chr = self.mutation(crossed)

            # 新一代 = 满足条件子代 + 满足条件亲代凑齐M
            new_gen = []
            for new_X in self.decoder(new_X_chr):
                if self.if_meet_constraints(new_X):
                    new_gen.append(self.encoder([new_X])[0])
            if len(new_gen) < self.M:
                for j in range(self.M): # 保留亲代最优个体
                    if self.if_meet_constraints(self.X[self.rank[j]]):
                        new_gen.append(self.X_chr[self.rank[j]])
                        break
            while len(new_gen) < self.M: # 保留部分亲代凑齐M个体
                random_choose = randint(0,self.M-1)
                if self.if_meet_constraints(self.X[random_choose]):
                    new_gen.append(self.X_chr[random_choose])
            self.X_chr = new_gen
            self.X = self.decoder(self.X_chr)

# %%
if __name__ == '__main__':

    date = '20181201'

    # run
    time_start = time.time()
    ga = GeneticAlgorithm(M=10, gen=2, pool_num=300, date=date, use_AGA=True)
    ga.run()
    time_end = time.time()
    print('Time spent:',round((time_end - time_start) / 3600, 2), 'hours')
    # # plot
    # plt.plot(ga.history['f'])
    # plt.title('Fitness value')
    # plt.xlabel('Iter')
    # plt.show()

