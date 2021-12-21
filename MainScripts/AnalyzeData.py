import pingouin as pt
import pandas as pd
pd.set_option('display.max_columns', None)
from HelperObjects.Analyzer import Analyzer
import matplotlib.pyplot as plt

ALPHA = 0.05
analyzer = Analyzer()
data = analyzer.get_prepared_data()

zero_after_count = data.loc[data["size_after"] <= 1].shape[0]
zero_before_count = data.loc[data["size_before"] <= 1].shape[0]
filtered = data.loc[(data["size_after"] > 1) & (data["size_before"] > 1)]


ax1 = plt.subplot(1, 2, 1)
ax1.set_xlim(left = 0, right = 1)
ax1.hist(filtered['avg_dist_before'], color='blue', edgecolor='black', bins=100)
ax2 = plt.subplot(1, 2, 2, sharey=ax1, sharex=ax1)
ax2.hist(filtered['avg_dist_after'], color='blue', edgecolor='black', bins=100)
plt.show()

res = pt.ttest(filtered["avg_dist_before"], filtered["avg_dist_after"], paired=True)
print(res)
print(res["p-val"] < ALPHA)
