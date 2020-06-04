import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def read_datafile(file):
    with open(file, 'r') as datafile:
        contents = datafile.read()
        data = np.array([line.split('\t') for line in contents.split('\n')], dtype=np.float64)
    return data


def read_data(data_str):
    data_sm = np.loadtxt(f"data/{data_str}_SM.txt", delimiter=',')
    data_class_labels = np.loadtxt(f"data/{data_str}_Class_Labels.txt", dtype='i')
    return data_sm, data_class_labels


def normalize_data(data_sm):
    min_data_sm = np.nanmin(data_sm)
    max_data_sm = np.nanmax(data_sm)
    return (data_sm - min_data_sm)/(max_data_sm - min_data_sm), min_data_sm, max_data_sm


def print_results(data_str, eer, frr, far, eer_threshold, eer_threshold_original, frr_at_10, frr_at_1, frr_at_01):
    print(f"{data_str}")
    print(f"EER:{eer:.1f}% "
          f"(FRR:{frr:.1f}%, FAR:{far:.1f}%)")
    print(f"Threshold that gives EER (in 0-1 range):{eer_threshold:.3f}")
    print(f"Threshold that gives EER (in original dataset scale):{eer_threshold_original:.2f}")
    print(f"FRR at FAR=10%\t:{frr_at_10:.1f}%")
    print(f"FRR at FAR=1%\t:{frr_at_1:.1f}%")
    print(f"FRR at FAR=0.1%\t:{frr_at_01:.1f}%")


def plot_score_distributions(data_str, data_ix, genuine_scores_list, impostor_scores_list, zoomed=False, axis_values=None):
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    sns.distplot(genuine_scores_list, hist=False, kde_kws={"color": "g", "lw": 3, "label": "Genuine"})
    sns.distplot(impostor_scores_list, hist=False, kde_kws={"color": "r", "lw": 3, "label": "Impostor"})
    if zoomed:
        plt.axis(axis_values)
        plt.title(f'Genuine and Impostor Score Distributions for {data_str} Dataset (Zoomed In)')
    else:
        plt.title(f'Genuine and Impostor Score Distributions for {data_str} Dataset')
    plt.xlabel('Score')
    plt.ylabel('Density')
    if zoomed:
        plt.savefig(f'output/scores_dataset{data_ix}_zoom.png')
    else:
        plt.savefig(f'output/scores_dataset{data_ix}.png')
    plt.show()


def plot_roc_curve(data_str, data_ix, far_list, frr_list, zoomed=False, axis_values=None):
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    plt.plot(np.array(far_list) / 100, 1 - np.array(frr_list) / 100, label='ROC Curve', linewidth=3)
    if zoomed:
        plt.axis(axis_values)
        plt.title(f'ROC curve for {data_str} Dataset (Zoomed In)')
    else:
        plt.title(f'ROC curve for {data_str} Dataset')
    plt.legend(loc='lower right')
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('Genuine Acceptance Rate')
    if zoomed:
        plt.savefig(f'output/roc_dataset{data_ix}_zoom.png')
    else:
        plt.savefig(f'output/roc_dataset{data_ix}.png')
    plt.show()
