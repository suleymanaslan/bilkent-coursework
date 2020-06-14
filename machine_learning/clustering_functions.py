import time
import numpy as np
from matplotlib import cm
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


def compute_distance_matrix(dataset, init_nan=False):
    sm = np.array([np.square(dataset - dataset[i]).sum(axis=1) for i in range(len(dataset))])
    sm[np.identity(len(sm)) == 1] = np.nan if init_nan else np.inf
    return sm


def merge_clusters_single_linkage(distance_matrix, clusters, len_dm):
    argmin_dm = np.argmin(distance_matrix)
    min_i, min_j = argmin_dm // len_dm, argmin_dm % len_dm
    cluster_min_i, cluster_min_j = clusters[min_i], clusters[min_j]
    if not cluster_min_i == cluster_min_j:
        cluster_eq_i = clusters == cluster_min_i
        cluster_eq_j = clusters == cluster_min_j
        temp_dm = distance_matrix[cluster_eq_i]
        temp_dm[:, cluster_eq_j] = np.inf
        distance_matrix[cluster_eq_i] = temp_dm    
        temp_dm = distance_matrix[cluster_eq_j]
        temp_dm[:, cluster_eq_i] = np.inf
        distance_matrix[cluster_eq_j] = temp_dm
        clusters[cluster_eq_j] = cluster_min_i
    else:
        print("Error")
    return distance_matrix, clusters


def merge_clusters_complete_linkage(distance_matrix, clusters, len_dm):
    nanargmin_dm = np.nanargmin(distance_matrix)
    min_i, min_j = nanargmin_dm // len_dm, nanargmin_dm % len_dm
    cluster_min_i, cluster_min_j = clusters[min_i], clusters[min_j]
    if not cluster_min_i == cluster_min_j:
        cluster_eq_i = clusters == cluster_min_i
        cluster_eq_j = clusters == cluster_min_j
        temp_dm = distance_matrix[cluster_eq_i]
        temp_dm[:, cluster_eq_j] = np.nan
        distance_matrix[cluster_eq_i] = temp_dm    
        temp_dm = distance_matrix[cluster_eq_j]
        temp_dm[:, cluster_eq_i] = np.nan
        distance_matrix[cluster_eq_j] = temp_dm
        clusters[cluster_eq_j] = cluster_min_i
        cluster_eq_i = clusters == cluster_min_i
        cluster_eq_j = clusters == cluster_min_j
        temp_dm = distance_matrix[cluster_eq_i]
        temp_dm[:] = np.max(temp_dm, axis=0)
        distance_matrix[cluster_eq_i] = temp_dm
        temp_dm = distance_matrix[:,cluster_eq_i]
        temp_dm[:] = np.expand_dims(np.max(temp_dm, axis=1), axis=1)
        distance_matrix[:,cluster_eq_i] = temp_dm
    else:
        print("Error")
    return distance_matrix, clusters


def merge_clusters_group_average(distance_matrix, clusters, len_dm):
    nanargmin_dm = np.nanargmin(distance_matrix)
    min_i, min_j = nanargmin_dm // len_dm, nanargmin_dm % len_dm
    cluster_min_i, cluster_min_j = clusters[min_i], clusters[min_j]
    if not cluster_min_i == cluster_min_j:
        cluster_eq_i = clusters == cluster_min_i
        cluster_eq_j = clusters == cluster_min_j
        temp_dm = distance_matrix[cluster_eq_i]
        temp_dm[:, cluster_eq_j] = np.nan
        distance_matrix[cluster_eq_i] = temp_dm    
        temp_dm = distance_matrix[cluster_eq_j]
        temp_dm[:, cluster_eq_i] = np.nan
        distance_matrix[cluster_eq_j] = temp_dm
        clusters[cluster_eq_j] = cluster_min_i
        cluster_eq_i = clusters == cluster_min_i
        cluster_eq_j = clusters == cluster_min_j
        temp_dm = distance_matrix[cluster_eq_i]
        temp_dm[:] = np.mean(temp_dm, axis=0)
        distance_matrix[cluster_eq_i] = temp_dm
        temp_dm = distance_matrix[:,cluster_eq_i]
        temp_dm[:] = np.expand_dims(np.mean(temp_dm, axis=1), axis=1)
        distance_matrix[:,cluster_eq_i] = temp_dm
    else:
        print("Error")
    return distance_matrix, clusters


def hierarchical_clustering(mode, dataset, nb_of_clusters, dataname="data", create_anim_file=False, plot_every_iter=15, color_list=None, print_output=True):
    start_time = time.time()
    if mode == "single":
        dm = compute_distance_matrix(dataset)
    elif mode == "complete" or "group_average":
        dm = compute_distance_matrix(dataset, init_nan=True)
    clusters = np.arange(len(dataset))
    len_dm = len(dm)
    
    hsv_colors = cm.get_cmap('hsv')
    
    for i in range(len(dataset) - nb_of_clusters):
        if mode == "single":
            dm, clusters = merge_clusters_single_linkage(dm, clusters, len_dm)
        elif mode == "complete":
            dm, clusters = merge_clusters_complete_linkage(dm, clusters, len_dm)
        elif mode == "group_average":
            dm, clusters = merge_clusters_group_average(dm, clusters, len_dm)
        
        unique_clusters = np.unique(clusters)
        if create_anim_file and (i % plot_every_iter == 0 or len(unique_clusters) < 8):
            if len(unique_clusters) < 8:
                color_arr = [color_list[np.where(unique_clusters == cluster)[0][0]] for cluster in clusters]
            else:
                color_arr = [hsv_colors(cluster/len(dataset)) for cluster in clusters]
            plt.scatter(dataset[:,0], dataset[:,1], c=color_arr)
            plt.title(f'{dataname}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f'gif/{dataname}_{i:04d}.png')
            plt.close()
    
    squared_errors = 0
    for i in unique_clusters:
        cluster_centroid = dataset[clusters == i].mean(axis=0)
        squared_error = np.square(dataset[clusters == i] - cluster_centroid).sum()
        squared_errors += squared_error
    
    end_time = time.time()
    if print_output:
        if mode == "single":
            print(f"Sum of squared errors for {dataname} (normalized) with single-linkage:{squared_errors:.4f}")
            print(f"Single-linkage for {dataname} took :{end_time - start_time:.3f} seconds")
        elif mode == "complete":
            print(f"Sum of squared errors for {dataname} (normalized) with complete-linkage:{squared_errors:.4f}")
            print(f"Complete-linkage for {dataname} took :{end_time - start_time:.3f} seconds")
        elif mode == "group_average":
            print(f"Sum of squared errors for {dataname} (normalized) with group average:{squared_errors:.4f}")
            print(f"Group average for {dataname} took :{end_time - start_time:.3f} seconds")
    
    if create_anim_file:
        if mode == "single":
            anim_file = f'gif/singlelinkage_{dataname}.gif'
        elif mode == "complete":
            anim_file = f'gif/completelinkage_{dataname}.gif'
        elif mode == "group_average":
            anim_file = f'gif/groupaverage_{dataname}.gif'
        png_files = f'gif/{dataname}_*.png'
        
        utils.create_animation(anim_file, png_files)


def single_linkage(dataset, nb_of_clusters, dataname="data", create_anim_file=False, plot_every_iter=15, color_list=None, print_output=True):
    hierarchical_clustering("single", dataset, nb_of_clusters, dataname, create_anim_file, plot_every_iter, color_list, print_output)


def complete_linkage(dataset, nb_of_clusters, dataname="data", create_anim_file=False, plot_every_iter=15, color_list=None, print_output=True):
    hierarchical_clustering("complete", dataset, nb_of_clusters, dataname, create_anim_file, plot_every_iter, color_list, print_output)


def group_average(dataset, nb_of_clusters, dataname="data", create_anim_file=False, plot_every_iter=15, color_list=None, print_output=True):
    hierarchical_clustering("group_average", dataset, nb_of_clusters, dataname, create_anim_file, plot_every_iter, color_list, print_output)
