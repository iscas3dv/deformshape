clear all; close all; clc
addpath(genpath('./train_data/50002/mesh_for_geodesic_distance/'))
addpath(genpath('./../Tools/'))

path_shapes = './train_data/50002/mesh_for_geodesic_distance/';
path_distance_matrix = './train_data/50002/distance_matrix/';
num_workers = 16;

calc_dist_matrix_shape_collection(path_shapes,path_distance_matrix,num_workers);