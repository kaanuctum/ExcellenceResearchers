import pingouin as pt
from HelperObjects.Analyzer import Analyzer
analyzer = Analyzer()
data = analyzer.get_prepared_data()

zero_after_count = data.loc[data["size_after"] <= 1].shape[0]
zero_before_count = data.loc[data["size_before"] <= 1].shape[0]

filtered = data.loc[(data["size_after"] > 1) & (data["size_before"] > 1)]


print(filtered.describe())

res = pt.ttest(filtered["avg_dist_before"], filtered["avg_dist_after"], paired=True)
print(res)