#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import numpy as np
import argparse
from utils_si import *
import sys
from copy import deepcopy

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RCSLS for supervised word alignment')

parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument("--src_mat", type=str, default='', help="To resume from an existing alignment. Load source alignment matrix. If none given, the aligment matrix is the identity.")

parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')

parser.add_argument("--dico_train", type=str, default='', help="train dictionary")
parser.add_argument("--dico_test", type=str, default='', help="validation dictionary")

parser.add_argument("--output", type=str, default='', help="where to save aligned embeddings")

parser.add_argument("--knn", type=int, default=10, help="number of nearest neighbors in RCSL/CSLS")
parser.add_argument("--maxneg", type=int, default=200000, help="Maximum number of negatives for the Extended RCSLS")
parser.add_argument("--maxsup", type=int, default=-1, help="Maximum number of training examples")
parser.add_argument("--maxload", type=int, default=200000, help="Maximum number of loaded vectors")

parser.add_argument("--model", type=str, default="none", help="Set of constraints: spectral or none")
parser.add_argument("--reg", type=float, default=0.0 , help='regularization parameters')

parser.add_argument("--lr", type=float, default=1.0, help='learning rate')
parser.add_argument("--niter", type=int, default=10, help='number of iterations')
parser.add_argument('--sgd', action='store_true', help='use sgd')
parser.add_argument("--batchsize", type=int, default=10000, help="batch size for sgd")
parser.add_argument('--patience', type=int, default=5, help='wiat this many iterations before earling stop')

parser.add_argument('--verbose', action='store_true', help='enable verbose')

params = parser.parse_args()

###### SPECIFIC FUNCTIONS ######
# functions specific to RCSLS
# the rest of the functions are in utils.py

def getknn(sc, x, y, k=10):
    print("Finding KNN scores")
    sidx = np.argpartition(sc, -k, axis=1)[:, -k:]
    ytopk = y[sidx.flatten(), :]
    ytopk = ytopk.reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
    f = np.sum(sc[np.arange(sc.shape[0])[:, None], sidx])
    df = np.dot(ytopk.sum(1).T, x)
    return f / k, df / k


def rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, knn=10):
    # print("Finding RCSLS..")
    Y_tgt_T = Y_tgt.T
    Z_tgt_T = Z_tgt.T
    R_T = R.T
    print("Finding RCSLS error")
    print("Calculating the transoformation")
    X_trans = np.dot(X_src, R_T)
    print("Finding f")
    f = 2 * np.sum(X_trans * Y_tgt)
    print(f"Finding df. Y_tgt_T: {Y_tgt_T.shape} -- X_src: {X_src.shape}")
    df = np.zeros((Y_tgt_T.shape[0], X_src.shape[1]))
    df = 2 * np.dot(Y_tgt_T, X_src)
    '''
    # df = 2 * (Y_tgt_T @ X_src)
    # df = 2 * np.matmul(Y_tgt_T, X_src)
    for i in range(Y_tgt_T.shape[0]):
        # df[i] = np.dot(Y_tgt_T[i], X_src)
        for j in range(X_src.shape[1]):
            print(i, j)
            print("Y_tgt_T[i]: ", Y_tgt_T[i])
            print("X_src[:,j]: ", X_src[:,j])
            df[i][j] = np.dot(Y_tgt_T[i], X_src[:,j])
    '''
    
    print(f"X_trans: {X_trans.shape}, Z_tgt_T: {Z_tgt_T.shape}")
    mat1 = np.zeros((X_trans.shape[0], Z_tgt_T.shape[1]))

    for i in range(X_trans.shape[0]):
        # print(i)
        mat1[i] = np.dot(X_trans[i], Z_tgt_T)

    # for i in range(Z_tgt_T.shape[1]):
    #     mat1[:, i] = np.dot(X_trans, Z_tgt_T[:, i])

    print("Mat1 done...")
    
    fk0, dfk0 = getknn(mat1, X_src, Z_tgt, knn)

    # fk0, dfk0 = getknn(np.dot(X_trans, Z_tgt_T), X_src, Z_tgt, knn)

    print(f"X_trans: {Z_src.shape}, Z_tgt_T: {R_T.shape}, Y_tgt_T: {Y_tgt_T.shape}")
    '''
    mat2 = np.zeros((Z_src.shape[0], R_T.shape[1]))
    
    for i in range(Z_src.shape[0]):
        mat2[i] = np.dot(Z_src[i], R_T)

    fk1, dfk1 = getknn(np.dot(mat2, Y_tgt_T).T, Y_tgt, Z_src, knn)
    '''
    fk1, dfk1 = getknn(np.dot(np.dot(Z_src, R_T), Y_tgt_T).T, Y_tgt, Z_src, knn)
    
    
    f = f - fk0 -fk1
    df = df - dfk0 - dfk1.T
    return -f / X_src.shape[0], -df / X_src.shape[0]


def proj_spectral(R):
    U, s, V = np.linalg.svd(R)
    s[s > 1] = 1
    s[s < 0] = 0
    return np.dot(U, np.dot(np.diag(s), V))


def load_transform(fname, d1=300, d2=300):
    print("Loading the alignment/transformation matrix...")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    try:
        d1, d2 = map(int, fin.readline().split())
    except:
        print("Embedding size is not specified.. Using default embedding sizes..")

    R = np.zeros([d1, d2])
    for i, line in enumerate(fin):
        tokens = line.split(' ')
        R[i, :] = np.array(tokens[0:d2], dtype=float)
    print("loading done!") 
    return R

###### MAIN ######

# load word embeddings
words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center, norm=True)
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center, norm=True)

# load validation bilingual lexicon
src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)
train_src2tgt, train_lexicon_size = load_lexicon(params.dico_train, words_src, words_tgt)

# word --> vector indices
idx_src = idx(words_src)
idx_tgt = idx(words_tgt)

# load train bilingual lexicon
pairs = load_pairs(params.dico_train, idx_src, idx_tgt)
if params.maxsup > 0 and params.maxsup < len(pairs):
    pairs = pairs[:params.maxsup]

# selecting training vector  pairs
X_src, Y_tgt = select_vectors_from_pairs(x_src, x_tgt, pairs)

# adding negatives for RCSLS
Z_src = x_src[:params.maxneg, :]
Z_tgt = x_tgt[:params.maxneg, :]

resumed = False

# initialization:
if params.src_mat != "":
    try:
        print("[RESUMING] Appliying given alignment to the src embeddings")
        R = load_transform(params.src_mat)
        resumed = True
    except:
        R = procrustes(X_src, Y_tgt)
    #print("[RESUMING] Appliying given alignment to the src embeddings")
    #R = load_transform(params.src_mat)
else:
    R = procrustes(X_src, Y_tgt)

# nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size, norm=False, norm_src_only=True)
nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=-1, norm=False, norm_src_only=True)
train_nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, train_src2tgt, lexicon_size=-1, norm=False, norm_src_only=True)
print(f"[init -- {'Procrustes' if not resumed else 'checkpoint mat'}] Test >> NN: %.4f"%(nnacc))
print(f"[init -- {'Procrustes' if not resumed else 'checkpoint mat'}] Train >> NN: %.4f"%(train_nnacc))
sys.stdout.flush()

# optimization
fold, Rold = 0, []
niter, lr = params.niter, params.lr

train_loss = {}
train_acc = {}
test_acc = {}

best_R = deepcopy(R)
best_nnacc = nnacc
n_no_improve = 0

for it in range(0, niter + 1):
    if lr < 1e-4:
        break

    if params.sgd:
        indices = np.random.choice(X_src.shape[0], size=params.batchsize, replace=False)
        f, df = rcsls(X_src[indices, :], Y_tgt[indices, :], Z_src, Z_tgt, R, params.knn)
    else:
        f, df = rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, params.knn)

    if params.reg > 0:
        R *= (1 - lr * params.reg)
    R -= lr * df
    if params.model == "spectral":
        R = proj_spectral(R)

    print("[it=%d] f = %.4f" % (it, f))
    sys.stdout.flush()

    if f > fold and it > 0 and not params.sgd:
        lr /= 2
        f, R = fold, Rold

    fold, Rold = f, R

    # if (it > 0 and it % 5 == 0) or it == niter:
    # nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size, norm=False, norm_src_only=True)
    nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=-1, norm=False, norm_src_only=True)
    train_nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, train_src2tgt, lexicon_size=-1, norm=False, norm_src_only=True)
    print("[it=%d] Test >> NN = %.4f - Coverage = %.4f" % (it, nnacc, len(src2tgt) / lexicon_size))
    print("[it=%d] Train >> NN = %.4f - Coverage = %.4f" % (it, train_nnacc, len(train_src2tgt) / train_lexicon_size))
    print()

    train_loss[it] = f
    train_acc[it] = train_nnacc
    test_acc[it] = nnacc

    if best_nnacc < nnacc:
        print(f"New best alignment found with test NN acc: {best_nnacc:.4f} --> {nnacc:.4f}")
        best_nnacc = nnacc
        best_R = deepcopy(R)
        n_no_improve = 0
    else:
        n_no_improve += 1

    if n_no_improve >= params.patience:
        print(f"Stopping the training since no improvement on the testset for {n_no_improve} iterations...")
        print(f"Best test NN acc reported: {best_nnacc:.4f}, current acc: {nnacc:.4f}")
        break

R = best_R

# nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size, norm=False, norm_src_only=True)
nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=-1, norm=False, norm_src_only=True)
train_nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, train_src2tgt, lexicon_size=-1, norm=False, norm_src_only=True)
print("[final] Test >> NN = %.4f - Coverage = %.4f" % (nnacc, len(src2tgt) / lexicon_size))
print("[final] Train >> NN = %.4f - Coverage = %.4f" % (train_nnacc, len(train_src2tgt) / train_lexicon_size))

if params.output != "":
    print("Loading src embeddings to form aligned vectors")
    words_full, x_full = load_vectors(params.src_emb, maxload=-1, center=params.center, verbose=False)
    print("Aligning src embeddings")
    x = np.dot(x_full, R.T)
    print("Normalizaing aligned embeddings")
    # x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    
    for i in range(len(x)):
        x[i] /= np.linalg.norm(x[i]) + 1e-8
    
    print("Saving all aligned vectors at %s" % params.output)
    save_vectors(params.output, x, words_full)
    print("Saving all aligned matrix at %s-mat" % params.output)
    save_matrix(params.output + "-mat",  R)


plt.plot(train_acc.keys(), train_acc.values(), label="Train NN_acc")
plt.plot(test_acc.keys(), test_acc.values(), label="Test NN_acc")
plt.xlabel("Epoch")
plt.ylabel("NN_Acc")
plt.legend()

plt.savefig(f"{params.output.rsplit('/')[0]}/acc_{params.output.rsplit('/')[-1]}.png")

plt.figure()
plt.plot(train_loss.keys(), train_loss.values(), label="Train RCSLS_Loss")
plt.xlabel("Epoch")
plt.ylabel("RCSLS_Loss")
plt.legend()

plt.savefig(f"{params.output.rsplit('/')[0]}/loss_{params.output.rsplit('/')[-1]}.png")
