from HelperObjects.Analyzer import Analyzer, pathManager

import pingouin as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import linregress


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
    ax1.set_xlabel('Average Distance')
    ax1.set_ylabel('Number of Researchers')
    ax2.set_xlabel('Average Distance')
    ax2.set_ylabel('Number of Researchers')
    plt.tight_layout()
    res = pt.ttest(filtered["avg_dist_after"], filtered["avg_dist_before"], paired=True)
    res['Type'] = model
    print(res)
    print(res["p-val"] < alpha)
    plt.savefig(f'AnalyzeData/{model}_Distance.png')
    res.to_csv(f'AnalyzeData/{model}_t_test.csv')

    # look at the two effects seperatly
    filtered_bigger = filtered.loc[(filtered["avg_dist_after"] > filtered["avg_dist_before"])]
    filtered_smaller = filtered.loc[(filtered["avg_dist_after"] < filtered["avg_dist_before"])]

    res_bigger = pt.ttest(filtered_bigger["avg_dist_after"], filtered_bigger["avg_dist_before"], paired=True)
    res_smaller = pt.ttest(filtered_smaller["avg_dist_after"], filtered_smaller["avg_dist_before"], paired=True)

    res_smaller['Method'] = 'Smaller'
    res_bigger['Method'] = 'Bigger'

    res_separate = pd.concat([res_smaller, res_bigger])
    res_separate.to_csv(f'AnalyzeData/{model}t_test_separate.csv')

    ax1.cla()
    ax2.cla()
    bin_size = int(len(filtered_bigger) ** (1 / 2))
    ax1.hist(filtered_bigger['avg_dist_before'], color='blue', edgecolor='black', bins=bin_size)
    ax1.title.set_text('Average Distance Before')

    ax2.hist(filtered_bigger['avg_dist_after'], color='blue', edgecolor='black', bins=bin_size)
    ax2.title.set_text('Average Distance After')
    ax1.set_xlabel('Average Distance')
    ax1.set_ylabel('Number of Researchers')
    ax2.set_xlabel('Average Distance')
    ax2.set_ylabel('Number of Researchers')
    plt.tight_layout()

    plt.savefig(f'AnalyzeData/{model}_Distance_bigger.png')

    ax1.cla()
    ax2.cla()
    bin_size = int(len(filtered_smaller) ** (1 / 2))
    ax1.hist(filtered_smaller['avg_dist_before'], color='blue', edgecolor='black', bins=bin_size)
    ax1.title.set_text('Average Distance Before')
    ax2.hist(filtered_smaller['avg_dist_after'], color='blue', edgecolor='black', bins=bin_size)
    ax2.title.set_text('Average Distance After')
    ax1.set_xlabel('Average Distance')
    ax1.set_ylabel('Number of Researchers')
    ax2.set_xlabel('Average Distance')
    ax2.set_ylabel('Number of Researchers')
    plt.tight_layout()
    plt.savefig(f'AnalyzeData/{model}_Distance_smaller.png')

    return res


def compare_models():
    df = pd.DataFrame(pickle.load(open(pathManager.get_paths()['model_results'], 'rb')))
    df["Coherence_umass"] = -df["Coherence_umass"]
    ax = plt.gca()
    df.plot(kind='line', x='Topics', y="Coherence_cv", ax=ax)
    df.plot(kind='line', x='Topics', y="Coherence_umass", ax=ax.twinx(), color='red')
    plt.xlabel("Topic Number")
    plt.ylabel('Coherence Score')
    plt.savefig('AnalyzeData/models.png')


def combined_t_test():
    umass = analyze_before_after('UMASS')
    cv = analyze_before_after('CV')
    total = pd.concat([umass, cv])
    total.insert(1, 'Topics', [45, 31])
    total.set_index('Type')
    total.to_csv(f'AnalyzeData/combined_t_test.csv')


def analyze_per_year(model):
    analyzer = Analyzer(model)
    df = analyzer.calc_average_for_every_year()
    df.plot(y=['average'])
    plt.ylabel("Average Distance Between Documents")
    plt.xlabel('Published Years')
    plt.savefig(f"AnalyzeData/{model.upper()}_yearly_average.png")

    df = df.transpose()
    df['null_count'] = df.isnull().sum(axis=1)

    print(f"More than 35 published years: {df.loc[(df['null_count'] < (55 - 35))].shape[0]}")
    print(f"More than 40 published years: {df.loc[(df['null_count'] < (55 - 40))].shape[0]}")


def graph_article_counts():
    analyzer = Analyzer('UMASS')
    data = analyzer.main()
    data['total'] = data['size_before'] + data['size_after']
    bin_count = int(np.sqrt(data.shape[0]))

    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2, sharey=ax1, sharex=ax1)
    ax3 = plt.subplot(2, 3, 3, sharey=ax1, sharex=ax1)

    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5, sharey=ax4, sharex=ax4)
    ax6 = plt.subplot(2, 3, 6, sharey=ax4, sharex=ax4)

    cut_off = 1000

    ax1.set_xlim(0, cut_off)
    ax2.set_xlim(0, cut_off)
    ax3.set_xlim(0, cut_off)

    ax1.hist(data['size_before'], color='blue', edgecolor='black', bins=bin_count)
    ax1.title.set_text('Before Grant')
    ax1.set_ylabel('Researcher Count')
    ax1.set_xlabel('Publication Count')

    ax2.hist(data['size_after'], color='blue', edgecolor='black', bins=bin_count)
    ax2.title.set_text('After Grant')
    ax2.set_ylabel('Researcher Count')
    ax2.set_xlabel('Publication Count')

    ax3.hist(data['total'], color='blue', edgecolor='black', bins=bin_count)
    ax3.title.set_text('Total Count')
    ax3.set_ylabel('Researcher Count')
    ax3.set_xlabel('Publication Count')

    cut_off = 300

    ax4.set_xlim(0, cut_off)
    ax5.set_xlim(0, cut_off)
    ax6.set_xlim(0, cut_off)

    bin_count = int(np.sqrt(data.loc[data["total"] < cut_off].shape[0]) * 3)

    ax4.hist(data['size_before'], color='blue', edgecolor='black', bins=bin_count)
    ax4.title.set_text('Before Grant\nZoomed')
    ax4.set_ylabel('Researcher Count')
    ax4.set_xlabel('Publication Count')

    ax5.hist(data['size_after'], color='blue', edgecolor='black', bins=bin_count)
    ax5.title.set_text('After Grant\nZoomed')
    ax5.set_ylabel('Researcher Count')
    ax5.set_xlabel('Publication Count')

    ax6.hist(data['total'], color='blue', edgecolor='black', bins=bin_count)
    ax6.title.set_text('Total Count\nZoomed')
    ax6.set_ylabel('Researcher Count')
    ax6.set_xlabel('Publication Count')

    plt.tight_layout(pad=0.4, w_pad=1, h_pad=2)
    plt.savefig('AnalyzeData/publication_counts.png')


def yearly_averages_for_authors(model):
    analyzer = Analyzer(model)
    output = analyzer.normalized_distance_to_years()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = ax1.twinx()
    filtered = output.loc[output['dt'] > -51]
    filtered.plot(kind='line', x='dt', y="count", ax=ax2)  # number of researchers with >1 researchers
    filtered.plot(kind='line', x='dt', y="avg_dist", ax=ax1, color='red')
    ax1.set_xlabel("Years after getting the grant (-1: before)")
    ax1.set_ylabel('researcher count')
    ax2.set_ylabel('average distance in year')
    ax2.set_ylim(0, 1750)
    ax1.set_ylim(0.25, 0.5)
    ax1.vlines(x=0, ymin=-1, ymax=1, linestyles='dashed', color="black")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    filtered = output.loc[output['dt'] > -31]
    ax3 = plt.subplot(2, 1, 2)
    ax4 = ax3.twinx()
    filtered.plot(kind='line', x='dt', y="count", ax=ax4)  # number of researchers with >1 researchers
    filtered.plot(kind='line', x='dt', y="avg_dist", ax=ax3, color='red')
    ax3.set_xlabel("Years after getting the grant (-1: before)")
    ax3.set_ylabel('researcher count')
    ax4.set_ylabel('average distance in year')
    ax4.set_ylim(0, 1750)
    ax3.set_ylim(0.35, 0.45)
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')
    ax3.vlines(x=0, ymin=-1, ymax=1, linestyles='dashed', color="black")

    '''
    reg_total = linregress(filtered['dt'].values, filtered['avg_dist'].values)
    reg_before = linregress(filtered.loc[filtered['dt'] < 0]['dt'].values, filtered.loc[filtered['dt'] < 0]['avg_dist'].values)
    reg_after = linregress(filtered.loc[filtered['dt'] >= 0]['dt'].values, filtered.loc[filtered['dt'] >= 0]['avg_dist'].values)
    ax.axline(xy1=(0, reg_total.intercept), slope=reg_total.slope, linestyle="--", color="black")
    ax.axline(xy1=(0, reg_before.intercept), slope=reg_before.slope, linestyle="--", color="red")
    ax.axline(xy1=(0, reg_after.intercept), slope=reg_after.slope, linestyle="--", color="blue")
    '''
    plt.tight_layout()

    plt.savefig('AnalyzeData/normalized_averages.png')


def normalized_comp_temp():
    df = pd.DataFrame
    ax = plt.gca()
    ax2 = ax.twinx()
    df.plot(kind='bar', x='dt', y="count", ax=ax)
    df.plot(kind='line', x='dt', y="avg", ax=ax2, color='red')
    ax.set_xlabel("Years after getting the grant (-1: before)")
    ax.set_ylabel('researcher count')
    ax2.set_ylabel('average distance in year')
    plt.savefig('AnalyzeData/models.png')


if __name__ == "__main__":
    # graph_article_counts()
    # compare_models()
    # analyze_per_year('umass')
    # analyze_before_after('cv')
    # analyze_before_after("umass")
    yearly_averages_for_authors('UMASS')
