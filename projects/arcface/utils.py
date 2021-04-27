"""
Classes and functions for face validation.
"""

import os
import torch
import math
import numpy as np

from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from tqdm import tqdm

from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

    
class PairDataset(Dataset):
    def __init__(self,
            data_root: str = None,
            pair_list: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader
        ) -> None:
        super(PairDataset, self).__init__()
        self.data_root = data_root
        self.pair_list = pair_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.make_dataset()

    def make_dataset(self) -> None:
        with open(self.pair_list, 'r') as f:
            self.pairs = [e for e in f]
        print(f'=> load {len(self.pairs)} test pairs')

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path1, path2, target = self.pairs[index].strip().split()
        sample1 = self.loader(os.path.join(self.data_root, path1))
        sample2 = self.loader(os.path.join(self.data_root, path2))
        target = int(target)

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, sample2, target

    def __len__(self) -> int:
        return len(self.pairs)

    
def distance(embeddings0, embeddings1, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings0, embeddings1)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
        norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
        # shaving
        similarity = np.clip(dot / norm, -1., 1.)
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_same == 0:
        val = 0
    else:
        val = float(true_accept) / float(n_same)
    if n_diff == 0:
        far = 0
    else:
        far = float(false_accept) / float(n_diff)
    return val, far


def calculate_roc(thresholds, embeddings0, embeddings1, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings0.shape[0] == embeddings1.shape[0])
    assert(embeddings0.shape[1] == embeddings1.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings0.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)
    pbar = tqdm(enumerate(k_fold.split(indices)), total=nrof_folds)
    for fold_idx, (train_set, test_set) in pbar:
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings0[train_set], embeddings1[train_set]]), axis=0)
        else:
            mean = 0.

        dist = distance(embeddings0-mean, embeddings1-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        # report progress
        pbar.set_description(f"fold iter: {fold_idx + 1} / {nrof_folds}")

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_val(thresholds, embeddings0, embeddings1, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings0.shape[0] == embeddings1.shape[0])
    assert(embeddings0.shape[1] == embeddings1.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings0.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    pbar = tqdm(enumerate(k_fold.split(indices)), total=nrof_folds)
    for fold_idx, (train_set, test_set) in pbar:
        # why to do it here
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings0[train_set], embeddings1[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings0-mean, embeddings1-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

        # report progress
        pbar.set_description(f"fold iter: {fold_idx + 1} / {nrof_folds}")

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def evaluate(embeddings0, embeddings1, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    print('=> running calculate_roc ...')
    tpr, fpr, accuracy = calculate_roc(
            thresholds, embeddings0, embeddings1, actual_issame,
            nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean
            )

    thresholds = np.arange(0, 4, 0.001)
    print('=> running calculate_val ...')
    val, val_std, far = calculate_val(
            thresholds, embeddings0, embeddings1, actual_issame, 1e-3,
            nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean
            )

    # results
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)

    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr, bounds_error=False, fill_value=(1., 0.))(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)

    return tpr, fpr, accuracy, val, val_std, far
