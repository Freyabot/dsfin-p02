#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算实际数据统计量
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# 设置项目根目录
import os
os.chdir(r"d:\2026研一下\dsfin-hw\dshw-p02")

# 股票列表
stocks = [
    {"code": "sh.600036", "name": "招商银行", "industry": "银行"},
    {"code": "sh.601328", "name": "交通银行", "industry": "银行"},
    {"code": "sz.002594", "name": "比亚迪", "industry": "汽车"},
    {"code": "sh.600104", "name": "上汽集团", "industry": "汽车"},
    {"code": "sz.000002", "name": "万科A", "industry": "房地产"},
    {"code": "sh.600048", "name": "保利发展", "industry": "房地产"},
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sz.000858", "name": "五粮液", "industry": "白酒"},
    {"code": "sh.601012", "name": "隆基绿能", "industry": "能源"},
    {"code": "sh.600028", "name": "中国石化", "industry": "能源"}
]

print("--- 加载数据 ---")
combined_df = pd.read_csv("data/combined/combined_data.csv")
combined_df['date'] = pd.to_datetime(combined_df['date'])

# === 4.1 基本统计量 ===
print("\n--- 计算日收益率描述性统计 ---")

def calculate_max_drawdown(prices):
    """计算最大回撤"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

stats_list = []

for stock in stocks:
    stock_data = combined_df[combined_df['name'] == stock['name']].copy()

    if 'return' not in stock_data.columns or stock_data['return'].isna().all():
        stock_data = stock_data.sort_values('date')
        stock_data['return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))

    returns = stock_data['return'].dropna()

    if len(returns) > 0:
        annual_mean = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        max_dd = calculate_max_drawdown(stock_data.set_index('date')['close'])

        stats_list.append({
            '股票': stock['name'],
            '行业': stock['industry'],
            '年化均值': annual_mean,
            '年化波动率': annual_vol,
            '偏度': skewness,
            '峰度': kurtosis,
            '最大回撤': max_dd
        })

stats_df = pd.DataFrame(stats_list)
print("\n收益率描述性统计:")
print(stats_df.round(4).to_string(index=False))

# === CAPM模型 ===
print("\n--- CAPM模型估计 ---")

# 准备数据
market_df = combined_df.groupby('date')['沪深300_close'].first().reset_index()
market_df = market_df.sort_values('date')
market_df['market_return'] = np.log(market_df['沪深300_close'] / market_df['沪深300_close'].shift(1))
market_df = market_df.dropna()

# 无风险利率（年化2%，日频）
rf_daily = 0.02 / 252

capm_results = []
beta_results = []

for stock in stocks:
    stock_data = combined_df[combined_df['name'] == stock['name']].copy()
    stock_data = stock_data.sort_values('date')

    if 'return' not in stock_data.columns or stock_data['return'].isna().all():
        stock_data['return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))

    # 合并市场数据
    reg_data = pd.merge(stock_data[['date', 'return']],
                        market_df[['date', 'market_return']],
                        on='date', how='inner')
    reg_data = reg_data.dropna()

    if len(reg_data) > 30:
        # 计算超额收益
        reg_data['excess_return'] = reg_data['return'] - rf_daily
        reg_data['excess_market'] = reg_data['market_return'] - rf_daily

        # OLS回归
        X = sm.add_constant(reg_data['excess_market'])
        model = sm.OLS(reg_data['excess_return'], X).fit()

        alpha = model.params['const']
        alpha_pval = model.pvalues['const']
        beta = model.params['excess_market']
        beta_ci = model.conf_int().loc['excess_market'].values
        r2 = model.rsquared

        capm_results.append({
            '股票': stock['name'],
            '行业': stock['industry'],
            'alpha': alpha,
            'alpha_pval': alpha_pval,
            'beta': beta,
            'beta_ci_low': beta_ci[0],
            'beta_ci_high': beta_ci[1],
            'r2': r2
        })

capm_df = pd.DataFrame(capm_results)
print("\nCAPM回归结果汇总:")
display_cols = ['股票', '行业', 'alpha', 'alpha_pval', 'beta', 'beta_ci_low', 'beta_ci_high', 'r2']
print(capm_df[display_cols].round(4).to_string(index=False))

# === 宏观数据相关分析 ===
print("\n--- 宏观指标分析 ---")

# 计算沪深300月度收益率
hs300_daily = combined_df.groupby('date')['沪深300_close'].first().reset_index()
hs300_daily = hs300_daily.sort_values('date')
hs300_daily['month'] = pd.to_datetime(hs300_daily['date']).dt.to_period('M')
hs300_monthly = hs300_daily.groupby('month').last().reset_index()
hs300_monthly['monthly_return'] = np.log(hs300_monthly['沪深300_close'] / hs300_monthly['沪深300_close'].shift(1))

# 获取宏观数据（月度）
macro_monthly = combined_df.groupby('year_month').agg({
    'cpi': 'first',
    'm2': 'first'
}).reset_index()
macro_monthly['month'] = pd.to_datetime(macro_monthly['year_month'], format='%Y-%m').dt.to_period('M')

# 合并数据
analysis_df = pd.merge(hs300_monthly[['month', 'monthly_return']],
                        macro_monthly[['month', 'cpi', 'm2']],
                        on='month', how='inner')
analysis_df = analysis_df.dropna()

if len(analysis_df) > 2:
    r_value, p_value = stats.pearsonr(analysis_df['cpi'].dropna(), analysis_df['monthly_return'].dropna())
    print(f"\nCPI与沪深300月度收益率的Pearson相关系数: {r_value:.4f} (p值: {p_value:.4f})")

# === 相关系数矩阵 ===
print("\n--- 收益率相关系数 ---")
return_wide = combined_df.pivot(index='date', columns='name', values='return')
corr_matrix = return_wide.corr()
print("\n相关系数矩阵（前5只股票）:")
print(corr_matrix.iloc[:5, :5].round(4))

# 保存统计结果
stats_df.to_csv('output/stats_summary.csv', index=False, encoding='utf-8-sig')
capm_df.to_csv('output/capm_results.csv', index=False, encoding='utf-8-sig')
print("\n统计结果已保存到output/目录")
