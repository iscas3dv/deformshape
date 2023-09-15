This is the code of "Self-supervised Learning of Implicit Shape Representation with Dense Correspondence for Deformable Objects".

It is adapted from the code of [DIF](https://github.com/microsoft/DIF-Net). We appreciate their significant contributions to the field of shape representation.

# 1. Installation
## Clone this repository.

## Install dependencies.
1. create an environment
```
conda create -n deformshape python=3.9
conda activate deformshape
```
2. Install pytorch.
3. Install torchmeta. Before installation, comment [L34~35 in 
pytorch-meta/setup.py](https://github.com/tristandeleu/pytorch-meta/blob/d55d89ebd47f340180267106bde3e4b723f23762/setup.py#L34), which limits the pytorch vision. We have tested that it is compatible with higher pytorch version.
Additionally, comment out [line 3 in pytorch-meta/torchmeta/datasets/utils.py](https://github.com/tristandeleu/pytorch-meta/blob/794bf82348fbdc2b68b04f5de89c38017d54ba59/torchmeta/datasets/utils.py#L3).
```
cd pytorch-meta
python setup.py install
```
4. Install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
5. Install [torchmcubes](https://github.com/tatsy/torchmcubes).
6. Install packages
```
pip install -r requirements.txt
```

# 2.Preparation
We use [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf) to generate SDFs from meshes.
Please follow the instructions to install mesh-to-sdf package first.
The experiments are conducted mainly on [D-FAUST](https://dfaust.is.tue.mpg.de/) and [DeformingThings4D](https://github.com/rabbityl/DeformingThings4D).
Please download these two datasets and organize the datasets as follows:
```
D-FAUST
│
└─── registrations_f.hdf5
│
└─── registrations_m.hdf5
```
```
DeformingThings4D
│
└─── animals
...
```
First, please run this code to generate train/eval data for DeformingThings4D:
```
python data/dataprocess_DeformingThings4D.py --dataset_folder $DeformingThings4D_folder_path$ 
--output_folder data/train_data(or test_data) --animal_kind bear3EP --mode train(or test)
```
Then, the preparing data are organized as follows:
```
train_data
│
└─── bear3EP
│   │
|   └─── surface_pts_n_normal
|   |   |
|   |   └─── *.mat
│   |
|   └─── free_space_pts
|   |   |
|   |   └─── *.mat
|   |
|   └─── mesh
|   |   |
|   |   └─── *.obj
|   | 
|   └─── mesh_for_geodesic_distance
|   |   |
|   |   └─── *.ply
|   |
|   └─── partial_points
|   |   |
|   |   └─── *.mat
...
```
Next, please run this code to generate train/eval data for D-FAUST:
```
python data/DFAUST_hdf5_to_obj.py --path $D-FAUST_path$/registrations_f.hdf5 --tdir data/D-FAUST_mesh
python data/DFAUST_hdf5_to_obj.py --path $D-FAUST_path$/registrations_m.hdf5 --tdir data/D-FAUST_mesh
python data/dataprocess_DFAUST.py --dataset_folder data/D-FAUST_mesh --output_folder data/train_data(or test_data)
--human_kind 50002 --mode train(or test)
```
Then, the preparing data are organized as follows:
```
train_data
│
└─── 50002
│   │
|   └─── surface_pts_n_normal
|   |   |
|   |   └─── *.mat
│   |
|   └─── free_space_pts
|   |   |
|   |   └─── *.mat
|   |
|   └─── mesh
|   |   |
|   |   └─── *.obj
|   | 
|   └─── mesh_for_geodesic_distance
|   |   |
|   |   └─── *.ply
|   |
|   └─── partial_points
|   |   |
|   |   └─── *.mat
...
```
The geodesic distances are only used for evaluation followed the steps. It takes a long time to generate geodesic distance matrix.
If you want to evaluate the correspondence result, you could follow the steps below to generate distance matrix.
The original matlab code is provided by [Unsupervised Learning Of Dense Shape Correspondence](https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence)
  

1.copy ```data/convert_ply_to_mat.m``` to```data/train_data/$subject$(eg, 50002)/mesh_for_geodesic_distance```

2.run ```data/train_data/$subject$(eg, 50002)/mesh_for_geodesic_distance/convert_ply_to_mat.m```

3.open ```data/faust_calc_distance_matrix.m```, change ```path_shapes, path_distance_matrix```, then run the code ```data/faust_calc_distance_matrix.m```

4.the distance matrix will be generated in 3.make a new directory in ```data/train_data/$subject$(eg, 50002)/distance_matrix```   
# 3. Training
Before training, set the `point_cloud_path` in the config files. For example,
```
point_cloud_root: data/train_data/
```
Then run the code:
```
python train.py --config configs/train/50002.yml
```
The code will make a folder `logs` and the parameters of trained model will be saved in it.
# 4. Shape Generation
## Generating shapes in training set
Directly use the latent code optimzed during training to generate a shapes:
```
python generate.py --config configs/generate_all/50002.yml --subject_idx 0,1,2,-1
```
subject_idx is the shape id in training split. `-1` represents template code. We found that `--subject_idx -1` is invalid for `configargparse`. There must be a id or ids before `-1`.
## Generating all shapes in training set
Before generateing, set the checkpoint path in the corresponding config files. For example,
```
checkpoint_path: 'logs/50002_train/model_final.pth'
```
Then run the code:
```
python evaluate.py --config configs/generate_all/50002.yml --type generate
```
Optimized results will be saved in `eval`

## Fitting all shapes from full observed point clouds
```
python evaluate.py --config configs/eval/50002.yml --type fit
```
Optimized results will be saved in `eval`

## Fitting all shapes from partial observed point clouds
```
python evaluate.py --config configs/eval_partial/50002.yml --type fit
```
Optimized results will be saved in `eval`


# 5. Accuracy
## Chamfer distance
```
python evaluate_chamfer.py
```
## IoU
```
python evaluate_iou_multiprocess.py
```

## Correspondence
```
python evaluate_corr.py
```