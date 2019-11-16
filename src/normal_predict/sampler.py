import torch
import os
import sys
sys.path.append(os.path.expanduser('~/Workspace/libigl/python'))
import pyigl as igl
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import geom_utils
from utils import utils_pt as util
from models import DirDeepModel, LapDeepModel, IdDeepModel, AvgModel, MlpModel, LapMATModel, GatDeepModel, EfficientCascade
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import torch.nn.functional as F
import random
import re
import glob
import utils.mesh as mesh
from iglhelpers import e2p, p2e
from sklearn.externals import joblib


"""
def calc_L2(L):
    ones = np.ones_like(L.data)
    adj = sp.sparse.coo_matrix((np.ones_like(L.data), (L.row, L.col)), shape=L.shape)
    D = 1.0 / np.squeeze(np.sqrt(np.asarray(adj.sum(1))))
    D[np.isinf(D)] = 0
    D = sp.sparse.diags(D, 0)
    L = D * L * D
    return L * L
"""


def read_npz(seq_names, args):
    with open(seq_names) as fp:
        Xd, Xi = igl.eigen.MatrixXd, igl.eigen.MatrixXi
        eV, eF, eVN = Xd(), Xi(), Xd()
        igl.readOBJ(seq_names, eV,Xd(), eVN, eF, Xi(), Xi())

        """
        # Upsample
        for _ in range(args.upsample):
            eVN = Xd()
            igl.upsample(eV, eF)
            igl.per_vertex_normals(eV, eF, eVN)
        """

        new_frame = {}
        npfloat = np.float32
        V, F = e2p(eV), e2p(eF)
        # Fix Degen,
        VN = e2p(eVN).astype(npfloat)
        vdist = VN

        L, mass, Di, DiA, weight = None, None, None, None, None

        if not np.isfinite(vdist).all():
            print(f'warning: {seq_names} nan vdist')
            return None

        if 'hack1' in args.additional_opt:
            hack = 1
        elif 'hack0' in args.additional_opt:
            hack=0

        if 'intrinsic' in args.additional_opt:
            hack = None
        def hackit(Op, h):
            Op.data[np.where(np.logical_not(np.isfinite(Op.data)))[0]] = h
            Op.data[Op.data > 1e10] = h
            Op.data[Op.data < -1e10] = h
            return Op

        if args.uniform_mesh:
            V -= np.min(V, axis=0)
            V /= np.max(V) # isotropic scaling
        if args.model.startswith('dirac'):
            Di, DiA = geom_utils.dirac(V, F)
            Di = Di.astype(np.float32)
            DiA = DiA.astype(np.float32)
            Di = hackit(Di, hack)
            DiA = hackit(DiA, hack)
            Di, DiA = util.sp_sparse_to_pt_sparse(Di), util.sp_sparse_to_pt_sparse(DiA)
            new_frame['Di'] = Di
            new_frame['DiA'] = DiA
            if not (torch.isfinite(Di._values()).all() and torch.isfinite(DiA._values()).all()):
            # if np.isfinite(Di.data).all() and np.isfinite(DiA.data).all():
                print(f'warning: {seq_names} nan D')
                return None
        else:
            if L is None:
                if hack is None:
                    import ipdb;ipdb.set_trace()
                    L = mesh.intrinsic_laplacian(V,F)
                else:
                    L = geom_utils.hacky_compute_laplacian(V,F, hack)

            if L is None:
                print("warning: {} no L".format(seq_names))
                return None
            if np.any(np.isnan(L.data)):
                print(f"warning: {seq_names} nan L")
                return None

            # L = calc_L2(L)
            new_frame['L'] = util.sp_sparse_to_pt_sparse(L.astype(np.float32))

        input_tensors = {}
        if 'V' in args.input_type:
            input_tensors['V'] = V

        new_frame['input'] = torch.cat([torch.from_numpy(input_tensors[t]) for t in input_tensors ], dim=1)

        # save data to new frame
        new_frame['V'] = V
        new_frame['F'] = F
        new_frame['target_dist'] = torch.from_numpy(vdist).view(-1,3)
        new_frame['name'] = seq_names
        return new_frame

def generate_batch(samples, batch_size, max_vertices, in_dim, out_dim, device):
    inputs = torch.zeros(batch_size, max_vertices, in_dim)
    mask = torch.zeros(batch_size, max_vertices, 1)
    targets = torch.zeros(batch_size, max_vertices, out_dim)
    laplacian = []

    for b, sam in enumerate(samples):
        num_vertices, input_channel = sam['input'].shape
        inputs[b, : num_vertices, : input_channel] = sam['input']
        target_dist = sam['target_dist']
        mask[b, : num_vertices] = 1
        targets[b, :target_dist.shape[0], :out_dim] = target_dist

        L = sam['L']
        laplacian.append(L)

    laplacian = util.sparse_diag_cat(laplacian, max_vertices, max_vertices)
    laplacian = laplacian.to(device)
    mask = mask.to(device)

    return inputs.to(device), targets.to(device), mask, laplacian

def load_schedule(filename):
    with open(filename) as f:
        return [list(map(int, line.strip().split(','))) for line in f]

def produce_batch_from_schedule(schedule, all_samples):
    device = torch.device('cuda')
    for batch in schedule:
        max_vertices = 0
        samples = []
        in_dim = 0
        out_dim = 0
        sample_names = []
        for sample_id in batch:
            sam = all_samples[sample_id]
            samples.append(sam)
            num_vertices, in_dim = sam["input"].shape
            max_vertices = max(max_vertices, num_vertices)
            out_dim = sam["target_dist"].size(1)
            sample_names.append(sam['name'])
        yield generate_batch(samples, len(samples), max_vertices,
                             in_dim, out_dim, device), sample_names


def sample_batch(seq_names, args, num_batch):
    device = torch.device('cuda' if args.cuda else 'cpu')
    input_features = args.input_dim
    output_features = 0
    if args.shuffle:
        random.shuffle(seq_names)
    sample_id = 0
    num_samples = len(seq_names)

    batch_count = 0
    samples = []
    sample_names = []
    max_vertices = 0
    batch_size = 0

    while batch_count < num_batch:
        new_sample = None
        seq_choice = seq_names[sample_id]
        sample_id += 1
        if sample_id >= num_samples:
            sample_id = 0
        if isinstance(seq_choice, str) and os.path.isfile(seq_choice):
            new_sample = read_npz(seq_choice, args)
        else:
            assert args.pre_load
            new_sample = seq_choice
        assert(new_sample is not None)

        num_vertices, input_channel = new_sample['input'].shape

        if num_vertices > args.max_vertices:
            print("Sample ignored, "
                f"has {num_vertices} > {args.max_vertices} nodes")
            continue

        if args.var_size and args.use_threshold is not None:
            if max(max_vertices, num_vertices) * (batch_size + 1) > args.use_threshold:
                yield generate_batch(samples, batch_size, max_vertices, input_features, output_features, device), sample_names
                batch_count += 1
                samples = []
                sample_names = []
                batch_size = 0
                max_vertices = 0

        batch_size += 1
        samples.append(new_sample)
        sample_names.append(new_sample['name'])
        max_vertices = max(max_vertices, num_vertices)
        output_features = new_sample["target_dist"].size(1)

        if not args.var_size or args.use_threshold is None:
            if batch_size == args.batch_size:
                yield generate_batch(samples, batch_size, max_vertices, input_features, output_features, device), sample_names
                samples = []
                sample_names = []
                batch_size = 0
                max_vertices = 0
                batch_count += 1
