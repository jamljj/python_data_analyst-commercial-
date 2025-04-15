# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据导入
file_path = 'CDNOW_master.txt'
# 假设数据以空格分隔，你可以根据实际情况修改分隔符
# 读取数据并查看列名
import csv
with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    columns = next(reader)
    # 去除空字符串
    columns = [col for col in columns if col]   
    print('数据集的列名：', columns)    

df = pd.read_csv(file_path, delimiter=' ', on_bad_lines='skip')

# Ensure columns exist before processing
required_columns = ['dollar_value', 'number_of_cds', 'date', 'customer_id']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# 2. 数据清洗
# 处理缺失值
df = df.dropna()

# 处理重复值
df = df.drop_duplicates()

# 处理异常值（这里简单假设数值列的异常值为超出均值3倍标准差的值）
for col in ['dollar_value', 'number_of_cds']:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] <= mean + 3 * std) & (df[col] >= mean - 3 * std)]

# 3. 数据转换
# 将data列转换为日期格式
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# 4. 数据探索性分析
# 统计不同列的唯一值数量
unique_counts = df.nunique()
print('不同列的唯一值数量：')
print(unique_counts)

# 5. 对用户的购买行为分布进行分析
# 5.1 分析amount_product和num_product的分布情况
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df['dollar_value'], bins=20)
axes[0].set_title('dollar_value分布')
axes[1].hist(df['number_of_cds'], bins=20)
axes[1].set_title('number_of_cds分布')
plt.show()

# 5.2 计算不同时间段（按天）的购买次数和购买金额总和
if 'date' in df.columns:
    daily_purchase = df.groupby(df['date'].dt.date).agg({
        'customer_id': 'count',
        'amount_product': 'sum'
    })
    daily_purchase.columns = ['购买次数', '购买金额总和']
    print('按天统计的购买次数和购买金额总和：')
    print(daily_purchase)

# 6. 用户价值分析
# 6.1 计算每个用户的总消费金额和总购买数量
user_value = df.groupby('customer_id').agg({
    'dollar_value': 'sum',
    'number_of_cds': 'sum'
})
user_value.columns = ['总消费金额', '总购买数量']

# 6.2 对用户进行分层
# 假设按照总消费金额的高低分为高、中、低价值用户
quantiles = user_value['总消费金额'].quantile([0.25, 0.75])
low_value = user_value[user_value['总消费金额'] <= quantiles.loc[0.25]]
medium_value = user_value[(user_value['总消费金额'] > quantiles.loc[0.25]) & (user_value['总消费金额'] <= quantiles.loc[0.75])]
high_value = user_value[user_value['总消费金额'] > quantiles.loc[0.75]]

print('低价值用户数量：', len(low_value))
print('中价值用户数量：', len(medium_value))
print('高价值用户数量：', len(high_value))

# 7. 客单价现状分析
# 7.1 客单价总体分布
# 计算客单价
if 'dollar_value' in df.columns and 'number_of_cds' in df.columns:
    df['客单价'] = df['dollar_value'] / df['number_of_cds']
    plt.hist(df['客单价'], bins=20)
    plt.title('客单价分布')
    plt.show()

    # 计算平均客单价、中位数、标准差等统计量
    average_price = df['客单价'].mean()
    median_price = df['客单价'].median()
    std_price = df['客单价'].std()
    print('平均客单价：', average_price)
    print('客单价中位数：', median_price)
    print('客单价标准差：', std_price)

# 7.2 客单价的时间趋势
if 'date' in df.columns and '客单价' in df.columns:
    daily_avg_price = df.groupby(df['date'].dt.date)['客单价'].mean()
    plt.plot(daily_avg_price)
    plt.title('客单价按天的时间趋势')
    plt.show()

# 7.3 不同用户群体的客单价差异
# 这里简单按照购买频率和总消费金额进行分组
if 'customer_id' in df.columns and 'dollar_value' in df.columns:
    purchase_frequency = df.groupby('customer_id').size()
    total_consumption = df.groupby('customer_id')['dollar_value'].sum()
    frequency_quantiles = purchase_frequency.quantile(0.5)
    consumption_quantiles = total_consumption.quantile(0.5)
    df = df.merge(purchase_frequency.rename('购买频率'), on='customer_id')
    df = df.merge(total_consumption.rename('总消费金额'), on='customer_id')
    df['购买频率等级'] = np.where(df['购买频率'] > frequency_quantiles, '高频', '低频')
    df['总消费等级'] = np.where(df['总消费金额'] > consumption_quantiles, '高消费', '低消费')
    df['用户群体'] = df['购买频率等级'] + df['总消费等级']
    group_price = df.groupby('用户群体')['客单价'].mean()
    print('不同用户群体的客单价：')
    print(group_price)

# 8. 用户特征与客单价的关系
# 由于数据集中未明确提及年龄、性别、地域等信息，这里跳过相关分析

# 9. 时间因素与客单价的关系
# 假设按照工作日和周末进行分析
if 'date' in df.columns and '客单价' in df.columns:
    df['是否周末'] = df['date'].dt.weekday >= 5
    weekend_price = df.groupby('是否周末')['客单价'].mean()
    print('工作日和周末的客单价：')
    print(weekend_price)

# 10. 购买数量与客单价的关系
if 'number_of_cds' in df.columns and '客单价' in df.columns:
    plt.scatter(df['number_of_cds'], df['客单价'])
    plt.xlabel('购买数量')
    plt.ylabel('客单价')
    plt.title('购买数量与客单价的关系')
    plt.show()
