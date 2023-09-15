#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import argparse
import collections
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from utils_si import *

parser = argparse.ArgumentParser(description='Evaluation of word alignment')
parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
parser.add_argument("--src_mat", type=str, default='', help="Load source alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--tgt_mat", type=str, default='', help="Load target alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--dico_test", type=str, default='', help="test dictionary")
parser.add_argument("--fig_prefix", type=str, default='', help="test results plot save name prefix")
parser.add_argument("--maxload", type=int, default=200000, help="max no. of word vectors to consider for the evaluation")
parser.add_argument("--top_k", type=int, default=10, help="top-k value for accuracy evaluations")
parser.add_argument("--nomatch", action='store_true', help="no exact match in lexicon")

params = parser.parse_args()


def load_vectors(fname, maxload=200000, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v

    if norm:
        # x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
        for j in range(len(x)):
            x[j] /= np.linalg.norm(x[j]) + 1e-8
    if center:
        # x -= x.mean(axis=0)[np.newaxis, :]
        # x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
        for j in range(len(x)):
            x[j] -= x[j].mean()
            x[j] /= np.linalg.norm(x[j]) + 1e-8
    if verbose:
        print("%d word vectors loaded from %d" % (len(words), n))
    return words, x


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def load_lexicon(filename, words_src, words_tgt, swap_file_columns=False, return_raw_lexicon=False, verbose=True):
    """
    swap_file_columns flag is when first column of the filename should be treated as the target
    """

    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    raw_lexicon = []
    raw_lexicon_words = []
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        l_split = line.split()

        if swap_file_columns:
            word_tgt, word_src = l_split[0], " ".join(l_split[1:])
        else:
            word_src, word_tgt = l_split[0], " ".join(l_split[1:])

        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
            if return_raw_lexicon:
                raw_lexicon.append((idx_src[word_src], idx_tgt[word_tgt]))
                raw_lexicon_words.append((word_src, word_tgt))

        vocab.add(word_src)


    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))

    # print(raw_lexicon, len(raw_lexicon), "..............")
    # print(raw_lexicon_words, len(raw_lexicon_words), "-------------")

    return lexicon, float(len(vocab)), np.array(raw_lexicon)


def compute_nn_accuracy(x_src, x_tgt, src2tgt_lexicon, tgt2src_lexicon, raw_lexicon, bsz=100, top_k=1):
    print(f"[INFO] Calculating top-{top_k} NN accuracy for {len(raw_lexicon)} pairs with batch-size {bsz}...")
    coords = []
    heatmap = np.zeros((top_k, top_k))

    lexicon_size = len(raw_lexicon)
    # src2tgt_lexicon_size = len(src2tgt_lexicon)
    # tgt2src_lexicon_size = len(tgt2src_lexicon)

    # idx_src = list(src2tgt_lexicon.keys())
    # idx_tgt = list(tgt2src_lexicon.keys())
    # print(words_tgt, len(words_tgt))
    # exit(0)
    src_2_tgt_acc = 0.0
    tgt_2_src_acc = 0.0

    # x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_src)):
        x_src[i] /= np.linalg.norm(x_src[i]) + 1e-8

    # x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_tgt)):
        x_tgt[i] /= np.linalg.norm(x_tgt[i]) + 1e-8

    for i in range(0, len(raw_lexicon), bsz):
        
        if i % 1000 == 0:
            print(f"[INFO] Processing batch {i} element")

        e = min(i + bsz, len(raw_lexicon))

        src_2_tgt_scores = np.dot(x_tgt, x_src[raw_lexicon[i:e, 0]].T)
        tgt_2_src_scores = np.dot(x_src, x_tgt[raw_lexicon[i:e, 1]].T)

        # print(src_2_tgt_scores)
        # print("---------------------")
        # print(tgt_2_src_scores)
        # exit(0)

        src_2_tgt_pred = src_2_tgt_scores.argsort(axis=0)[-top_k:][::-1].T
        tgt_2_src_pred = tgt_2_src_scores.argsort(axis=0)[-top_k:][::-1].T

        # print(src_2_tgt_pred)
        # print("----------------")
        # print(tgt_2_src_pred)

        for j in range(i, e):
            # print(src_2_tgt_pred[j - i])
            # print("*************")
            # print(raw_lexicon[j])
            # print(raw_lexicon[j, 1])
            # print(raw_lexicon[j, 0])
            # exit(0)
            coord_tgt = None
            coord_src = None

            for k, one_top_k in enumerate(src_2_tgt_pred[j - i]):
                # print(f"{one_top_k} ({words_tgt[one_top_k] if len(words_tgt) < one_top_k else ''}) --> {raw_lexicon[j, 1]} ({words_tgt[raw_lexicon[j, 1]] if len(words_tgt) < raw_lexicon[j, 1] else ''})")
                # print(f"{one_top_k} ({words_tgt[one_top_k]}) --> {raw_lexicon[j, 1]} ({words_tgt[raw_lexicon[j, 1]]})")
                if one_top_k == raw_lexicon[j, 1]:
                    src_2_tgt_acc += 1.0
                    coord_tgt = k
                    # print(f"CSLS TGT Hit {one_top_k} -- {coord_tgt} ({words_tgt[one_top_k]})")
                    break

            for k, one_top_k in enumerate(tgt_2_src_pred[j - i]):
                # print(f"{one_top_k} **> {raw_lexicon[j, 0]}")
                # print(f"{one_top_k} ({words_src[one_top_k]}) --> {raw_lexicon[j, 0]} ({words_src[raw_lexicon[j, 0]]})")
                if one_top_k == raw_lexicon[j, 0]:
                    tgt_2_src_acc += 1.0
                    coord_src = k
                    # print(f"CSLS SRC Hit {one_top_k} -- {coord_src} ({words_src[one_top_k]})")
                    break
            
            if (coord_src is not None) and (coord_tgt is not None):
                # print(f"Adding coord: ({coord_tgt}, {coord_src})")
                noise_tgt = np.random.uniform(0, 0.5)
                noise_src = np.random.uniform(0, 0.5)
                coords.append((coord_tgt + noise_tgt, coord_src + noise_src))
                heatmap[coord_src][coord_tgt] += 1
    
    return src_2_tgt_acc / lexicon_size, tgt_2_src_acc / lexicon_size, np.array(coords), heatmap


# def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024, top_k=1):
def compute_csls_accuracy(x_src, x_tgt, src2tgt_lexicon, tgt2src_lexicon, raw_lexicon, k=10, bsz=1024, top_k=1):
    print(f"[INFO] Calculating top-{top_k} CSLS accuracy for {len(raw_lexicon)} pairs with batch-size {bsz}...")
    coords = []
    heatmap = np.zeros((top_k, top_k))

    lexicon_size = len(raw_lexicon)
    # src2tgt_lexicon_size = len(src2tgt_lexicon)
    # tgt2src_lexicon_size = len(tgt2src_lexicon)

    src_2_tgt_acc = 0.0
    tgt_2_src_acc = 0.0

    # x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_src)):
        x_src[i] /= np.linalg.norm(x_src[i]) + 1e-8

    # x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_tgt)):
        x_tgt[i] /= np.linalg.norm(x_tgt[i]) + 1e-8

    # tgt2src
    sr = x_tgt[list(raw_lexicon[:, 1])]
    print("Fininding init dot tgt2src")
    sc = np.dot(sr, x_src.T)
    print("Done init dot tgt2src")
    tgt2src_similarities = 2 * sc
    sc2 = np.zeros(x_src.shape[0])

    n_loops = int(x_src.shape[0] / bsz)
    for i_loop, i in enumerate(range(0, x_src.shape[0], bsz)):

        if i_loop % 10 == 0:
            print(f"[INFO] Processing loop {i_loop}/{n_loops}")

        j = min(i + bsz, x_src.shape[0])
        sc_batch = np.dot(x_src[i:j, :], x_tgt.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    
    print("Loop done tgt2src")
    tgt2src_similarities -= sc2[np.newaxis, :]

    print("Finding top-k matches tgt2src")
    tgt2src_nn = np.argsort(tgt2src_similarities, axis=1).T[-top_k:][::-1].T

    # src2tgt
    sr = x_src[list(raw_lexicon[:, 0])]
    print("Fininding init dot src2tgt")
    sc = np.dot(sr, x_tgt.T)
    print("Done init dot src2tgt")
    src2tgt_similarities = 2 * sc
    sc2 = np.zeros(x_tgt.shape[0])

    n_loops = int(x_tgt.shape[0] / bsz)
    for i_loop, i in enumerate(range(0, x_tgt.shape[0], bsz)):
        print(f"[INFO] Processing loop {i_loop}/{n_loops}")
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    
    print("Loop done src2tgt")
    src2tgt_similarities -= sc2[np.newaxis, :]

    print("Finding top-k matches src2tgt")
    src2tgt_nn = np.argsort(src2tgt_similarities, axis=1).T[-top_k:][::-1].T

    # # tgt2src
    # sr = x_tgt[list(raw_lexicon[:, 1])]
    # print("Fininding init dot tgt2src")
    # sc = np.dot(sr, x_src.T)
    # print("Done init dot tgt2src")
    # tgt2src_similarities = 2 * sc
    # sc2 = np.zeros(x_src.shape[0])

    # for i in range(0, x_src.shape[0], bsz):
    #     j = min(i + bsz, x_src.shape[0])
    #     sc_batch = np.dot(x_src[i:j, :], x_tgt.T)
    #     dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
    #     sc2[i:j] = np.mean(dotprod, axis=1)
    
    # print("Loop done tgt2src")
    # tgt2src_similarities -= sc2[np.newaxis, :]

    # print("Finding top-k matches tgt2src")
    # tgt2src_nn = np.argsort(tgt2src_similarities, axis=1).T[-top_k:][::-1].T
    
    print("Finding CSLS scores and plotting coordinates...")
    for k in range(0, len(raw_lexicon)):

        coord_tgt = None
        coord_src = None

        for m, one_top_k in enumerate(src2tgt_nn[k]):
            # print(f"{one_top_k} ({words_tgt[one_top_k]}) --> {raw_lexicon[k, 1]} ({words_tgt[raw_lexicon[k, 1]]})")
            if one_top_k == raw_lexicon[k, 1]:
                src_2_tgt_acc += 1.0
                coord_tgt = m
                # print(f"CSLS TGT Hit {one_top_k} -- {coord_tgt} ({words_tgt[one_top_k]})")
                break

        for m, one_top_k in enumerate(tgt2src_nn[k]):
            # print(f"{one_top_k} ({words_src[one_top_k]}) **> {raw_lexicon[k, 0]} ({words_src[raw_lexicon[k, 0]]})")
            if one_top_k == raw_lexicon[k, 0]:
                tgt_2_src_acc += 1.0
                coord_src = m
                # print(f"CSLS SRC Hit {one_top_k} -- {coord_src} ({words_src[one_top_k]})")
                break
        
        if (coord_src is not None) and (coord_tgt is not None):
            # print(f"Adding coord: ({coord_tgt}, {coord_src})")
            noise_tgt = np.random.uniform(0, 0.5)
            noise_src = np.random.uniform(0, 0.5)
            coords.append((coord_tgt + noise_tgt, coord_src + noise_src))
            heatmap[coord_src][coord_tgt] += 1
    
    return src_2_tgt_acc / lexicon_size, tgt_2_src_acc / lexicon_size, np.array(coords), heatmap


# def load_transform(fname, d1=300, d2=300):
def load_transform(fname): #, d1=300, d2=300):
    print("Loading the alignment/transformation matrix...")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    d1, d2 = map(int, fin.readline().split())
    R = np.zeros([d1, d2])
    for i, line in enumerate(fin):
        tokens = line.split(' ')
        R[i, :] = np.array(tokens[0:d2], dtype=float)
    return R


###### MAIN ######

print("\n===============================================")

print("Evaluation of alignment on %s" % params.dico_test)
if params.nomatch:
    print("running without exact string matches")

words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)

if params.tgt_mat != "":
    R_tgt = load_transform(params.tgt_mat)
    # x_tgt = np.dot(x_tgt, R_tgt)
    x_tgt = np.dot(x_tgt, R_tgt.T)
if params.src_mat != "":
    R_src = load_transform(params.src_mat)
    # x_src = np.dot(x_src, R_src)
    x_src = np.dot(x_src, R_src.T)

src2tgt, src2tgt_lexicon_size, raw_lexicon = load_lexicon(params.dico_test, words_src, words_tgt, return_raw_lexicon=True)
tgt2src, tgt2src_lexicon_size, _ = load_lexicon(params.dico_test, words_tgt, words_src, swap_file_columns=True)

# print(raw_lexicon)
# exit(0)

# nnacc = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
# cslsproc = compute_csls_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
src2tgt_nnacc, tgt2src_nnacc, nn_coords, nn_heatmap = compute_nn_accuracy(x_src, x_tgt, src2tgt, tgt2src, raw_lexicon, top_k=params.top_k)
print("SRC2TGT >>> NN = %.4f - Coverage = %.4f" % (src2tgt_nnacc, len(src2tgt) / src2tgt_lexicon_size))
print("TGT2SRC >>> NN = %.4f - Coverage = %.4f" % (tgt2src_nnacc, len(tgt2src) / tgt2src_lexicon_size))
# src2tgt_nnacc, tgt2src_nnacc = 0, 0
# src2tgt_cslsproc, tgt2src_cslsproc = 0, 0
src2tgt_cslsproc, tgt2src_cslsproc, csls_coords, csls_heatmap = compute_csls_accuracy(x_src, x_tgt, src2tgt, tgt2src, raw_lexicon, top_k=params.top_k)
# src2tgt_cslsproc = compute_csls_accuracy(x_src, x_tgt, src2tgt, tgt2src, raw_lexicon, top_k=params.top_k)
# print("SRC2TGT >>> NN = %.4f - CSLS = %.4f - Coverage = %.4f" % (src2tgt_nnacc, src2tgt_cslsproc, len(src2tgt) / src2tgt_lexicon_size))
# print("TGT2SRC >>> NN = %.4f - CSLS = %.4f - Coverage = %.4f" % (tgt2src_nnacc, tgt2src_cslsproc, len(tgt2src) / tgt2src_lexicon_size))

print("SRC2TGT >>> CSLS = %.4f - Coverage = %.4f" % (src2tgt_cslsproc, len(src2tgt) / src2tgt_lexicon_size))
print("TGT2SRC >>> CSLS = %.4f - Coverage = %.4f" % (tgt2src_cslsproc, len(tgt2src) / tgt2src_lexicon_size))

avg_nn_acc = np.round((src2tgt_nnacc + tgt2src_nnacc) / 2, 2)
avg_csls_acc = np.round((src2tgt_cslsproc + tgt2src_cslsproc) / 2, 2)

# print(f"Coords: {nn_coords}")

# exit(0)

src_mat_name = params.src_mat.rsplit("/")[-1]

plt.scatter(nn_coords[:, 0], nn_coords[:, 1], s=1)
plt.xticks(np.arange(0, params.top_k, 1))
plt.yticks(np.arange(0, params.top_k, 1))
plt.xlabel("src2tgt")
plt.ylabel("tgt2src")
plt.title(f"NN bidirectional top-{params.top_k} lookup acc-{avg_nn_acc}")
# plt.show()
plt.savefig(f"coords-{params.fig_prefix}_nn_top-{params.top_k}_maxload-{params.maxload}_src_mat-{src_mat_name}_alignment_results.png")

plt.figure()

plt.scatter(csls_coords[:, 0], csls_coords[:, 1], s=1, color="red")
plt.xticks(np.arange(0, params.top_k, 1))
plt.yticks(np.arange(0, params.top_k, 1))
plt.xlabel("src2tgt")
plt.ylabel("tgt2src")
plt.title(f"CSLS bidirectional top-{params.top_k} lookup acc-{avg_csls_acc}")
# plt.show()
plt.savefig(f"coords-{params.fig_prefix}_csls_top-{params.top_k}_maxload-{params.maxload}_src_mat-{src_mat_name}_alignment_results.png")


################ NN Heatmap
plt.figure()

hm = sns.heatmap(
    data=nn_heatmap,
    annot=True,
    fmt='g',
    xticklabels=[(i+1) for i in range(params.top_k)], 
    yticklabels=[(i+1) for i in range(params.top_k)]
)

hm.invert_yaxis()
plt.xlabel("src2tgt")
plt.ylabel("tgt2src")
plt.title(f"NN bidirectional top-{params.top_k} lookup acc-{avg_nn_acc}")
# plt.show()
plt.savefig(f"heatmap-{params.fig_prefix}_nn_top-{params.top_k}_maxload-{params.maxload}_src_mat-{src_mat_name}_alignment_results.png")

################ CSLS Heatmap
plt.figure()

hm = sns.heatmap(
    data=csls_heatmap,
    annot=True,
    fmt='g',
    xticklabels=[(i+1) for i in range(params.top_k)], 
    yticklabels=[(i+1) for i in range(params.top_k)]
)

hm.invert_yaxis()
plt.xlabel("src2tgt")
plt.ylabel("tgt2src")
plt.title(f"CSLS bidirectional top-{params.top_k} lookup acc-{avg_csls_acc}")
# plt.show()
plt.savefig(f"heatmap-{params.fig_prefix}_csls_top-{params.top_k}_maxload-{params.maxload}_src_mat-{src_mat_name}_alignment_results.png")
