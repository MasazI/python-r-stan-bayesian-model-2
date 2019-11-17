import seaborn as sns
import pandas
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ファイルの読み込み
salary = pandas.read_csv('data-salary.txt')

# データの先頭5行の確認
print(salary.head())

# データのサマリを確認
print(salary.describe())

# データの図示
sns.scatterplot(
    x='X',
    y='Y',
    data=salary
)
plt.show()

# 単回帰 Seaborn
sns.lmplot("X","Y",salary)
plt.show()

# 単回帰 sklearn
x = salary[['X']]
y = salary[['Y']]
model_lr = LinearRegression()
model_lr.fit(x, y)

# 係数
print(model_lr.coef_)
# 切片
print(model_lr.intercept_)
# 決定係数
print(model_lr.score(x, y))

