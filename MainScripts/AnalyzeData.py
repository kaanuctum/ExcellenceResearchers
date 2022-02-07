from HelperObjects.Analyzer import Analyzer, pathManager

import pingouin as pt
import pandas as pd
import matplotlib.pyplot as plt
import pickle

pd.set_option('display.max_columns', None)

def analyze_before_after(model):
    alpha = 0.05
    analyzer = Analyzer(model.upper())
    data = analyzer.main()

    zero_after_count = data.loc[data["size_after"] <= 1].shape[0]
    zero_before_count = data.loc[data["size_before"] <= 1].shape[0]
    filtered = data.loc[(data["size_after"] > 1) & (data["size_before"] > 1)]
    print(f"Zero before: {zero_before_count}\nZero after: {zero_after_count}")
    bin_size = int(len(filtered) ** (1 / 2))
    print(bin_size)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharey=ax1, sharex=ax1)
    ax1.set_xlim(left=0, right=1)

    ax1.hist(filtered['avg_dist_before'], color='blue', edgecolor='black', bins=bin_size)
    ax1.title.set_text('Average Distance Before')
    ax2.hist(filtered['avg_dist_after'], color='blue', edgecolor='black', bins=bin_size)
    ax2.title.set_text('Average Distance After')
    plt.show()

    res = pt.ttest(filtered["avg_dist_before"], filtered["avg_dist_after"], paired=True)
    print(res)
    print(res["p-val"] < alpha)
    plt.savefig(f'AnalyzeData/{model}_Distance.png')
    res.to_csv(f'AnalyzeData/{model}_t_test.csv')


def compare_models():
    df = pd.DataFrame(pickle.load(open(pathManager.get_paths()['model_results'], 'rb')))
    df["Coherence_umass"] = -df["Coherence_umass"]
    ax = plt.gca()
    df.plot(kind='line', x='Topics', y="Coherence_cv", ax=ax)
    df.plot(kind='line', x='Topics', y="Coherence_umass", ax=ax.twinx(), color='red')
    plt.savefig('AnalyzeData/models.png')


if __name__ == "__main__":
    compare_models()
    analyze_before_after('CV')
    analyze_before_after('UMASS')
