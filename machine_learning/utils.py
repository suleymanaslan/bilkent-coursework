import os
import glob
import imageio
import numpy as np
from matplotlib import pyplot as plt


def read_datafile(file):
    with open(file, 'r') as datafile:
        contents = datafile.read()
        data = np.array([line.split('\t') for line in contents.split('\n')], dtype=np.float64)
    return data


def read_dataset():
    train1_x, train1_y = read_datafile('data/train1').T
    train2_x, train2_y = read_datafile('data/train2').T
    
    test1_x, test1_y = read_datafile('data/test1').T
    test2_x, test2_y = read_datafile('data/test2').T
    
    return train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y


def read_dataset2():
    dataset1 = read_datafile('data/dataset1')
    dataset2 = read_datafile('data/dataset2')
    dataset3 = read_datafile('data/dataset3')
    print(f"dataset1:{dataset1.shape}, dataset2:{dataset2.shape}, dataset3:{dataset3.shape}")
    
    return dataset1, dataset2, dataset3


def normalize_data(data, training_mean, training_std):
    data = data - training_mean
    data = data / training_std
    return data


def normalize_data_dimensions(data):
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i] - np.mean(data[:,i])) / np.std(data[:,i])
    return data


def normalize_dataset(train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y):
    mean_train1_x = np.mean(train1_x)
    std_train1_x = np.std(train1_x)
    
    mean_train1_y = np.mean(train1_y)
    std_train1_y = np.std(train1_y)
    
    mean_train2_x = np.mean(train2_x)
    std_train2_x = np.std(train2_x)
    
    mean_train2_y = np.mean(train2_y)
    std_train2_y = np.std(train2_y)
    
    train1_x = normalize_data(train1_x, mean_train1_x, std_train1_x)
    train1_y = normalize_data(train1_y, mean_train1_y, std_train1_y)
    train2_x = normalize_data(train2_x, mean_train2_x, std_train2_x)
    train2_y = normalize_data(train2_y, mean_train2_y, std_train2_y)
    
    test1_x = normalize_data(test1_x, mean_train1_x, std_train1_x)
    test1_y = normalize_data(test1_y, mean_train1_y, std_train1_y)
    test2_x = normalize_data(test2_x, mean_train2_x, std_train2_x)
    test2_y = normalize_data(test2_y, mean_train2_y, std_train2_y)
    
    return train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y


def normalize_dataset2(dataset1, dataset2, dataset3):
    dataset1 = normalize_data_dimensions(dataset1)
    dataset2 = normalize_data_dimensions(dataset2)
    dataset3 = normalize_data_dimensions(dataset3)
    
    return dataset1, dataset2, dataset3


def plot_data(x, y, title):
    fig = plt.figure()
    fig.set_facecolor('w')
    plt.scatter(x, y)
    plt.title(f'{title} data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_data2(data, color_list, title):
    fig = plt.figure()
    fig.set_facecolor('w')
    plt.scatter(data[:,0], data[:,1], c=color_list[-1])
    plt.title(f'{title}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_dataset(train1_x, train1_y, train2_x, train2_y, test1_x, test1_y, test2_x, test2_y):
    plot_data(train1_x, train1_y, "train1")
    plot_data(train2_x, train2_y, "train2")
    plot_data(test1_x, test1_y, "test1")
    plot_data(test2_x, test2_y, "test2")


def plot_dataset2(dataset1, dataset2, dataset3, color_list):
    plot_data2(dataset1, color_list, "dataset1")
    plot_data2(dataset2, color_list, "dataset2")
    plot_data2(dataset3, color_list, "dataset3")


def get_shape(train1_x, train2_x, test1_x, test2_x):
    train1_nb_examples = train1_x.shape[0]
    train2_nb_examples = train2_x.shape[0]
    
    test1_nb_examples = test1_x.shape[0]
    test2_nb_examples = test2_x.shape[0]
    
    return train1_nb_examples, train2_nb_examples, test1_nb_examples, test2_nb_examples


def get_uniform_samples(train1_x, train1_nb_examples, train2_x, train2_nb_examples):
    min_train1_x = np.min(train1_x)
    max_train1_x = np.max(train1_x)
    train1_uniform_x_samples = np.linspace(min_train1_x, max_train1_x, train1_nb_examples)
    
    min_train2_x = np.min(train2_x)
    max_train2_x = np.max(train2_x)
    train2_uniform_x_samples = np.linspace(min_train2_x, max_train2_x, train2_nb_examples)
    
    return train1_uniform_x_samples, train2_uniform_x_samples


def plot_output(x, y, uniform_x, output, title):
    fig = plt.figure()
    fig.set_facecolor('w')
    plt.scatter(x, y)
    plt.plot(uniform_x, output, linewidth=3)
    plt.title(f'{title}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def create_animation(anim_file, png_files, fps=8):
    frames = []
    filenames = glob.glob(png_files)
    filenames = sorted(filenames)
    for i, filename in enumerate(filenames):
        frames.append(imageio.imread(filename))
    for i in range(10):
        frames.append(imageio.imread(filename))
    for f in filenames:
        os.remove(f)
    imageio.mimsave(anim_file, frames, 'GIF', fps=fps)
