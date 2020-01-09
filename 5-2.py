import numpy as np
import seaborn as sns
import pandas
import mcmc_tools
import matplotlib.pyplot as plt
from scipy.stats import norm

# ファイルの読み込み
attendance_2 = pandas.read_csv('data-attendance-2.txt')
print(attendance_2.head())
print(attendance_2.describe())
