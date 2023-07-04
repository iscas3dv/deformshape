# -*- coding: utf-8 -*-
# Script to write registrations as obj files
# Copyright (c) [2015] [Gerard Pons-Moll]

from argparse import ArgumentParser
from os import makedirs
from os.path import join, exists
import h5py
import sys
from tqdm import tqdm


def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def get_seq(sid):
    file = open("./data/subjects_and_sequences.txt", 'r')
    seq_list = []
    for line in file.readlines():
        if line.startswith(sid):
            seq_list.append(line.strip())
    return seq_list

if __name__ == '__main__':

    # Subject ids
    sids = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
    # Sequences available for each subject id are listed in scripts/subjects_and_sequences.txt

    parser = ArgumentParser(description='Save sequence registrations as obj')
    parser.add_argument('--path', type=str, default='../registrations_f.hdf5',
                        help='dataset path in hdf5 format')
    parser.add_argument('--tdir', type=str, default='./',
                        help='target directory')
    args = parser.parse_args()

    for sid in sids:
        seq_list = get_seq(sid)
        print("generate {} meshes".format(sid))
        for seq in tqdm(seq_list):
            with h5py.File(args.path, 'r') as f:
                if seq not in f.keys():
                    print('Sequence %s from subject %s not in %s' % (seq, sid, args.path))
                    f.close()
                    continue
                verts = f[seq][()].transpose([2, 0, 1])
                faces = f['faces'][()]

            tdir = join(args.tdir, seq)
            if not exists(tdir):
                makedirs(tdir)

            # Write to an obj file
            for iv, v in enumerate(verts):
                fname = join(tdir, '{}_{}.obj'.format(seq, iv))
                print('Saving mesh %s' % fname)
                write_mesh_as_obj(fname, v, faces)