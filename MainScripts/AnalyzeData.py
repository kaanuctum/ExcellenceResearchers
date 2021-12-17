from HelperObjects.Analyzer import Analyzer
analyzer = Analyzer()
data = analyzer.get_prepared_data()

zero_after_count = data.loc[data["size_after"] == 0].shape[0]
zero_before_count = data.loc[data["size_before"] == 0].shape[0]

filtered = data.loc[(data["size_after"] > 1) & (data["size_before"] > 1)]