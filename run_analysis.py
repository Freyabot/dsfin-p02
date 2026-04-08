#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行分析并生成图表
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置项目根目录
os.chdir(r"d:\2026研一下\dsfin-hw\dshw-p02")
print(f"当前工作目录: {os.getcwd()}")

# 确保output目录存在
os.makedirs("output", exist_ok=True)

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

# 行业颜色映射
industry_colors = {
    '银行': '#1f77b4',
    '汽车': '#ff7f0e',
    '房地产': '#2ca02c',
    '白酒': '#d62728',
    '能源': '#9467bd'
}

# === 加载数据 ===
print("--- 加载真实数据 ---")
combined_df = pd.read_csv("data/combined/combined_data.csv")
combined_df['date'] = pd.to_datetime(combined_df['date'])
print(f"数据形状: {combined_df.shape}")
print(f"日期范围: {combined_df['date'].min()} 至 {combined_df['date'].max()}")

# === 图1：归一化收盘价走势图 ===
print("\n--- 绘制图1：归一化收盘价走势图 ---")
close_wide = combined_df.pivot(index='date', columns='name', values='close')
first_valid = close_wide.first_valid_index()
normalized = close_wide.div(close_wide.loc[first_valid])

if '沪深300_close' in combined_df.columns:
    hs300 = combined_df.groupby('date')['沪深300_close'].first()
    normalized['沪深300'] = hs300 / hs300.loc[first_valid]

fig, ax = plt.subplots(figsize=(14, 8))
for stock in stocks:
    name = stock['name']
    if name in normalized.columns:
        ax.plot(normalized.index, normalized[name],
                label=name, color=industry_colors[stock['industry']],
                linewidth=1.5, alpha=0.8)

if '沪深300' in normalized.columns:
    ax.plot(normalized.index, normalized['沪深300'],
            label='沪深300', color='black', linewidth=3, linestyle='--')

ax.set_title('归一化收盘价走势图 (2020-01-01 = 1)', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('归一化价格', fontsize=12)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('output/fig1_normalized_prices.png', dpi=150, bbox_inches='tight')
plt.close()
print("图1已保存")

# === 图2：日收益率分布图 ===
print("\n--- 绘制图2：日收益率分布图 ---")
fig, axes = plt.subplots(2, 5, figsize=(18, 10))
axes = axes.flatten()

for idx, stock in enumerate(stocks):
    ax = axes[idx]
    stock_data = combined_df[combined_df['name'] == stock['name']].copy()

    if 'return' not in stock_data.columns or stock_data['return'].isna().all():
        stock_data = stock_data.sort_values('date')
        stock_data['return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))

    returns = stock_data['return'].dropna()

    if len(returns) > 0:
        sns.histplot(returns, kde=True, ax=ax, color=industry_colors[stock['industry']], bins=50)
        mu, std = stats.norm.fit(returns)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p * len(returns) * (xmax - xmin) / 50, 'k--', linewidth=2)
        ax.set_title(f"{stock['name']}", fontsize=12, fontweight='bold')
        ax.text(0.05, 0.95, f'均值={mu*100:.3f}%\n标准差={std*100:.3f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('日收益率')
        ax.set_ylabel('频数')

plt.tight_layout()
plt.savefig('output/fig2_return_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("图2已保存")

# === 图3：收益率相关系数热力图 ===
print("\n--- 绘制图3：收益率相关系数热力图 ---")
return_wide = combined_df.pivot(index='date', columns='name', values='return')

if return_wide.isna().all().any():
    close_wide = combined_df.pivot(index='date', columns='name', values='close')
    return_wide = np.log(close_wide / close_wide.shift(1))

sorted_names = sorted(stocks, key=lambda x: (x['industry'], x['name']))
sorted_names = [s['name'] for s in sorted_names]
return_wide_sorted = return_wide[sorted_names]
corr_matrix = return_wide_sorted.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title('股票日收益率相关系数热力图', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("图3已保存")

# === 图4：宏观指标与股市关系 ===
print("\n--- 绘制图4：宏观指标与股市关系 ---")
if '沪深300_close' in combined_df.columns:
    hs300_daily = combined_df.groupby('date')['沪深300_close'].first().reset_index()
    hs300_daily = hs300_daily.sort_values('date')
    hs300_daily['month'] = pd.to_datetime(hs300_daily['date']).dt.to_period('M')
    hs300_monthly = hs300_daily.groupby('month').last().reset_index()
    hs300_monthly['monthly_return'] = np.log(hs300_monthly['沪深300_close'] / hs300_monthly['沪深300_close'].shift(1))

if 'cpi' in combined_df.columns:
    macro_monthly = combined_df.groupby('year_month').agg({
        'cpi': 'first',
        'm2': 'first'
    }).reset_index()
    macro_monthly['month'] = pd.to_datetime(macro_monthly['year_month'], format='%Y-%m').dt.to_period('M')

analysis_df = pd.merge(hs300_monthly[['month', 'monthly_return']],
                        macro_monthly[['month', 'cpi', 'm2']],
                        on='month', how='inner')
analysis_df = analysis_df.dropna()

fig, ax = plt.subplots(figsize=(10, 7))
x = analysis_df['cpi']
y = analysis_df['monthly_return'] * 100

sns.scatterplot(x=x, y=y, s=100, alpha=0.7, ax=ax, color='steelblue')

r_value = None
if len(x) > 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    ax.plot(x, line, 'r--', linewidth=2, label=f'拟合线 (R={r_value:.2f})')

ax.set_xlabel('CPI同比增速 (%)', fontsize=12)
ax.set_ylabel('沪深300月度收益率 (%)', fontsize=12)
ax.set_title('CPI与沪深300月度收益率关系', fontsize=16, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/fig4_macro_stock_relation.png', dpi=150, bbox_inches='tight')
plt.close()
print("图4已保存")

# === 图5：CAPM Beta系数 ===
print("\n--- 绘制图5：CAPM Beta系数 ---")
market_df = combined_df.groupby('date')['沪深300_close'].first().reset_index()
market_df = market_df.sort_values('date')
market_df['market_return'] = np.log(market_df['沪深300_close'] / market_df['沪深300_close'].shift(1))
market_df = market_df.dropna()

rf_daily = 0.02 / 252
beta_results = []

for stock in stocks:
    stock_data = combined_df[combined_df['name'] == stock['name']].copy()
    stock_data = stock_data.sort_values('date')

    if 'return' not in stock_data.columns or stock_data['return'].isna().all():
        stock_data['return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))

    reg_data = pd.merge(stock_data[['date', 'return']],
                        market_df[['date', 'market_return']],
                        on='date', how='inner')
    reg_data = reg_data.dropna()

    if len(reg_data) > 30:
        reg_data['excess_return'] = reg_data['return'] - rf_daily
        reg_data['excess_market'] = reg_data['market_return'] - rf_daily

        X = sm.add_constant(reg_data['excess_market'])
        model = sm.OLS(reg_data['excess_return'], X).fit()

        beta = model.params['excess_market']
        beta_ci = model.conf_int().loc['excess_market'].values

        beta_results.append({
            '股票': stock['name'],
            '行业': stock['industry'],
            'beta': beta,
            'beta_low': beta_ci[0],
            'beta_high': beta_ci[1]
        })

beta_df = pd.DataFrame(beta_results)
beta_df_sorted = beta_df.sort_values(['行业', 'beta'])

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(beta_df_sorted))

for i, (idx, row) in enumerate(beta_df_sorted.iterrows()):
    color = industry_colors[row['行业']]
    ax.errorbar(row['beta'], i,
                xerr=[[row['beta'] - row['beta_low']],
                      [row['beta_high'] - row['beta']]],
                fmt='o', capsize=5, elinewidth=2, markersize=8, color=color)

ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Beta = 1')
ax.set_yticks(y_pos)
ax.set_yticklabels(beta_df_sorted['股票'], fontsize=11)
ax.set_xlabel('Beta系数', fontsize=12)
ax.set_title('CAPM Beta系数与95%置信区间', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=ind,
                          markerfacecolor=color, markersize=8)
                   for ind, color in industry_colors.items()]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig('output/fig5_capm_betas.png', dpi=150, bbox_inches='tight')
plt.close()
print("图5已保存")

print("\n=== 所有图表生成完成 ===")
