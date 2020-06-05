import numpy as np
import seaborn as sns
import pandas as pd
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


def read_speaker_data():
    features = pd.read_csv("data/features.csv")
    features.drop('Filename', axis=1, inplace=True)
    label_dict = {}
    count = 0
    for label in features['Label'].unique():
        label_dict[label] = count
        count += 1
    features["Label"] = features["Label"].apply(lambda x: label_dict[x])
    return features, label_dict


def shuffle_split_data(features, label_dict):
    for label_i in range(len(label_dict)):
        cur_features = features[features["Label"] == label_i]
        cur_features = cur_features.sample(frac=1)
        row_count = cur_features.shape[0]
        train_size = int(row_count * 0.6)
        development_size = int(row_count * 0.2)
        test_size = row_count - (train_size + development_size)
        cur_train_set = cur_features.iloc[0:train_size]
        cur_development_set = cur_features.iloc[train_size:train_size + development_size]
        cur_test_set = cur_features.iloc[train_size + development_size:]
        if label_i == 0:
            train_set = cur_train_set
            development_set = cur_development_set
            test_set = cur_test_set
        else:
            train_set = train_set.append(cur_train_set)
            development_set = development_set.append(cur_development_set)
            test_set = test_set.append(cur_test_set)
    train_dev_set = train_set.append(development_set)
    return train_set, development_set, test_set, train_dev_set


def get_datasets(train_set, development_set, test_set, train_dev_set):
    train_x = train_set.drop('Label', axis=1).to_numpy()
    train_y = train_set["Label"].to_numpy()
    
    development_x = development_set.drop('Label', axis=1).to_numpy()
    development_y = development_set["Label"].to_numpy()
    
    test_x = test_set.drop('Label', axis=1).to_numpy()
    test_y = test_set["Label"].to_numpy()
    
    train_dev_x = train_dev_set.drop('Label', axis=1).to_numpy()
    train_dev_y = train_dev_set["Label"].to_numpy()
    
    return train_x, train_y, development_x, development_y, test_x, test_y, train_dev_x, train_dev_y


def plot_scores(dist_matrix, dataset_y, section_name, section_label, label_dict):
    genuine_scores_list = None
    impostor_scores_list = None
    for i in range(len(label_dict)):
        cur_dm = dist_matrix[:,i]
        genuine_scores, impostor_scores = cur_dm[dataset_y == i], cur_dm[dataset_y != i]
        genuine_scores_list = genuine_scores if genuine_scores_list is None else np.append(genuine_scores_list, genuine_scores)
        impostor_scores_list = impostor_scores if impostor_scores_list is None else np.append(impostor_scores_list, impostor_scores)
    sns.distplot(genuine_scores_list, hist=False, kde_kws={"color": "g", "lw": 3, "label": "Genuine"})
    sns.distplot(impostor_scores_list, hist=False, kde_kws={"color": "r", "lw": 3, "label": "Impostor"})
    plt.title(f'Genuine and Impostor Score Distributions for {section_name}')
    plt.xlabel(f'Scores')
    plt.ylabel('Density of Distance')
    plt.savefig(f'output/speaker_scores_{section_label}.png')
    plt.show()


def plot_far_frr(frr_list, far_list, threshold_list, section_name, section_label):
    plt.plot(np.array(threshold_list), np.array(far_list), label='FAR', linewidth=3)
    plt.plot(np.array(threshold_list), np.array(frr_list), label='FRR', linewidth=3)
    plt.title(f'FAR against FRR for {section_name}')
    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig(f'output/speaker_far_frr_{section_label}.png')
    plt.show()
