import numpy as np
from matplotlib import pyplot as plt

import utils


def initialize_centroids(data, k):
    cluster_centroids = np.concatenate((np.random.uniform(low=data[:,0].min(), high=data[:,0].max(), size=k).reshape(k, 1),
                                        np.random.uniform(low=data[:,1].min(), high=data[:,1].max(), size=k).reshape(k, 1)),
                                       axis=1)
    return cluster_centroids


def assign_clusters(data, k, cluster_centroids, data_clusters=None):
    cluster_ids = np.arange(k)
    nb_of_points = data.shape[0]
    min_distances = np.full(nb_of_points, np.inf)
    if data_clusters is None:
        data_clusters = np.zeros(nb_of_points)
    prev_data_clusters = np.copy(data_clusters)
    for i in range(k):
        cur_distances = np.square(data - cluster_centroids[i]).sum(axis=1)
        if len(data_clusters[cur_distances < min_distances]) > 0:
            data_clusters[cur_distances < min_distances] = i
        min_distances = np.minimum(cur_distances, min_distances)
    return data_clusters, np.array_equal(data_clusters, prev_data_clusters), min_distances


def update_centroids(data, k, data_clusters, cluster_centroids):
    for i in range(k):
        if len(data[data_clusters == i]) > 0:
            cluster_centroids[i,:] = data[data_clusters == i].mean(axis=0)
    return cluster_centroids


def kmeans(data, k, dataname="data", create_anim_file=False, color_list=None, print_output=True):
    np.random.seed(550)
    
    kmeans_iterations = 0
    
    if create_anim_file:
        anim_iter = 0
        plt.scatter(data[:,0], data[:,1], c=color_list[-1])
        plt.title(f'{dataname}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'gif/{dataname}_{anim_iter:04d}.png')
        plt.close()
    
    cluster_centroids = initialize_centroids(data, k)
    
    if create_anim_file:
        anim_iter += 1
        plt.scatter(data[:,0], data[:,1], c=color_list[-1])
        for i in range(k):
            plt.scatter(cluster_centroids[:,0][i], cluster_centroids[:,1][i], s=250, marker="X", c=color_list[i], linewidth=3, 
                        edgecolors='black')
        plt.title(f'{dataname}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'gif/{dataname}_{anim_iter:04d}.png')
        plt.close()
    
    data_clusters, done, min_distances = assign_clusters(data, k, cluster_centroids)
    kmeans_iterations += 1
    
    if create_anim_file:
        anim_iter += 1
        for i in range(k):
            plt.scatter(data[data_clusters == i][:,0], data[data_clusters == i][:,1], c=color_list[i])
            plt.scatter(cluster_centroids[:,0][i], cluster_centroids[:,1][i], s=250, marker="X", c=color_list[i], linewidth=3, 
                        edgecolors='black')
        plt.title(f'{dataname}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'gif/{dataname}_{anim_iter:04d}.png')
        plt.close()
    
    while not done:
        cluster_centroids = update_centroids(data, k, data_clusters, cluster_centroids)
        
        if create_anim_file:
            anim_iter += 1
            for i in range(k):
                plt.scatter(data[data_clusters == i][:,0], data[data_clusters == i][:,1], c=color_list[i])
                plt.scatter(cluster_centroids[:,0][i], cluster_centroids[:,1][i], s=250, marker="X", c=color_list[i],
                            linewidth=3, edgecolors='black')
            plt.title(f'{dataname}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f'gif/{dataname}_{anim_iter:04d}.png')
            plt.close()
        
        data_clusters, done, min_distances = assign_clusters(data, k, cluster_centroids, data_clusters=data_clusters)
        kmeans_iterations += 1
        
        if create_anim_file:
            anim_iter += 1
            for i in range(k):
                plt.scatter(data[data_clusters == i][:,0], data[data_clusters == i][:,1], c=color_list[i])
                plt.scatter(cluster_centroids[:,0][i], cluster_centroids[:,1][i], s=250, marker="X", c=color_list[i], 
                            linewidth=3, edgecolors='black')
            plt.title(f'{dataname}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f'gif/{dataname}_{anim_iter:04d}.png')
            plt.close()
    
    if print_output:
        print(f"Kmeans for {dataname} finished with {kmeans_iterations} iterations")
        print(f"Sum of squared errors for {dataname} (normalized) with kmeans:{min_distances.sum():.4f}")
    
    if create_anim_file:
        anim_file = f'gif/kmeans_{dataname}.gif'
        png_files = f'gif/{dataname}_*.png'
        
        utils.create_animation(anim_file, png_files)
