"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from scipy import linalg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


#https://github.com/mseitzer/pytorch-fid/blob/3d604a25516746c3a4a5548c8610e99010b2c819/src/pytorch_fid/fid_score.py#L149
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')



    parser.add_argument('--log_dir', type=str, default='pointnet2_ssg_wo_normals', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')

    return parser.parse_args()


def test(model, sample_pcs=None, ref_pcs=None, batch_size=4, fid_feat_dim=256):
    classifier = model.eval()
    if sample_pcs is None:
        sample_pcs = torch.rand((40, 1024, 3)).cuda()
    sample_pcs = sample_pcs.transpose(2, 1)
    if ref_pcs is None:
        ref_pcs = torch.rand((40, 1024, 3)).cuda()
    ref_pcs = ref_pcs.transpose(2, 1)
    assert sample_pcs.shape[0] == sample_pcs.shape[0]
    data_num = sample_pcs.shape[0]
    ref_feats = torch.zeros((data_num, fid_feat_dim))
    sample_feats = torch.zeros((data_num, fid_feat_dim))

    for ref_b_start in tqdm(range(0, data_num, batch_size)):
        ref_b_end = min(data_num, ref_b_start + batch_size)
        ref_batch = ref_pcs[ref_b_start:ref_b_end]
        sample_batch = sample_pcs[ref_b_start:ref_b_end]
        _, _, ref_feat = classifier(ref_batch)
        _, _, sample_feat = classifier(sample_batch)
        ref_feats[ref_b_start:ref_b_end] = ref_feat
        sample_feats[ref_b_start:ref_b_end] = sample_feat

    ref_feats = ref_feats.cpu().numpy()
    sample_feats = sample_feats.cpu().numpy()
    ref_mu = np.mean(ref_feats, axis=0)
    ref_sigma = np.cov(ref_feats, rowvar=False)
    sample_mu = np.mean(sample_feats, axis=0)
    sample_sigma = np.cov(sample_feats, rowvar=False)

    fid = calculate_frechet_distance(ref_mu,ref_sigma,sample_mu, sample_sigma)
    return fid
    #FID



def test_(model):
    classifier = model.eval()
    points = torch.rand((4, 1024, 3)).cuda()
    points = points.transpose(2, 1)
    pred, _, fid_feat = classifier(points)
    print(pred.shape)
    print(fid_feat.shape)




def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()


    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print('loading model ', model_name)
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint_path = str(experiment_dir) + '/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path)
    print('loading checkpoint: ', checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test(classifier.eval())



if __name__ == '__main__':
    args = parse_args()
    main(args)
