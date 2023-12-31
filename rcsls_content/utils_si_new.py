#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import pickle
import numpy as np
import collections
from sinling import word_splitter


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

    # if center:
    #     print("Centering and Normalizing the vectors...")
    #     # x -= x.mean(axis=0)[np.newaxis, :]
    #     # x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    #     for j in range(len(x)):
    #         x[j] -= x[j].mean()
    #         x[j] /= np.linalg.norm(x[j]) + 1e-8
    
    # elif norm:
    #     print("Normalizing the vectors...")
    #     # x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    #     for j in range(len(x)):
    #         x[j] /= np.linalg.norm(x[j]) + 1e-8

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


def wi(words):
    w2i = {}
    i2w = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
            i2w[i] = w

    return w2i, i2w


def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in range(n):
        fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def save_matrix(fname, x):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in range(n):
        fout.write(" ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def procrustes(X_src, Y_tgt):
    U, s, V = np.linalg.svd(np.dot(Y_tgt.T, X_src))
    return np.dot(U, V)


def select_vectors_from_pairs(x_src, y_tgt, pairs):
    n = len(pairs)
    d = x_src.shape[1]
    x = np.zeros([n, d])
    y = np.zeros([n, d])
    for k, ij in enumerate(pairs):
        i, j = ij
        x[k, :] = x_src[i, :]
        y[k, :] = y_tgt[j, :]
    return x, y


def load_lexicon(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        l_split = line.split()
        word_src, word_tgt = l_split[0], " ".join(l_split[1:])
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))


def load_pairs(filename, idx_src, idx_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    pairs = []
    tot = 0
    for line in f:
        l_split = line.rstrip().split(' ')
        a, b = l_split[0], " ".join(l_split[1:])
        if not b:
            l_split = line.rstrip().split('\t')
            a, b = l_split[0], " ".join(l_split[1:])
        tot += 1
        if a in idx_src and b in idx_tgt:
            pairs.append((idx_src[a], idx_tgt[b]))
    if verbose:
        coverage = (1.0 * len(pairs)) / tot
        print("Found pairs for training: %d - Total pairs in file: %d - Coverage of pairs: %.4f" % (len(pairs), tot, coverage))
    return pairs


def lematized_match(ref, pred, w2i_tgt, i2w_tgt, si_splits, si_splits_file_path):
    # ref = lexicon[idx_src[j]]
    # pred = pred[j - i]

    if not isinstance(ref, list):
        ref = list(ref)

    if not isinstance(pred, list):
        pred = list(pred)

    # any(one_top_k in list(lexicon[idx_src[j]]) for one_top_k in pred[j - i])

    split_ref = []

    for x in ref:

      if x in si_splits:
        split_ref.append(si_splits[x])
      else:
        try:
          split = word_splitter.split(i2w_tgt[x])['base']
        except:
          continue

        try:
          split = w2i_tgt[split]
        except:
          ...

        split_ref.append(split)
        si_splits[x] = split

        with open(si_splits_file_path, 'wb') as f:
            pickle.dump(si_splits, f)

    split_pred = []

    for y in pred:

      x = y.item()

      if x in si_splits:
        split_pred.append(si_splits[x])
      else:
        try:
          split = word_splitter.split(i2w_tgt[x])['base']
        except:
          continue

        try:
          split = w2i_tgt[split]
        except:
          ...

        split_pred.append(split)
        si_splits[x] = split

        with open(si_splits_file_path, 'wb') as f:
            pickle.dump(si_splits, f)

    split_ref = ref + split_ref
    split_pred = pred + split_pred

    return any(one_top_k in split_ref for one_top_k in split_pred)


def compute_nn_accuracy(x_src, x_tgt, lexicon, bsz=100, lexicon_size=-1, top_k=1, norm=False, norm_src_only=False, \
    do_lemmatized_match=False, w2i_tgt=None, i2w_tgt=None, si_splits=None, si_splits_file_path=None):  # use norm=False if vectors are already normalized
    if lexicon_size < 0:
        lexicon_size = len(lexicon)

    idx_src = list(lexicon.keys())
    acc = 0.0

    # if norm_src_only or norm:
    # x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_src)):
        x_src[i] /= np.linalg.norm(x_src[i]) + 1e-8

    # if norm:
    # x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_tgt)):
        x_tgt[i] /= np.linalg.norm(x_tgt[i]) + 1e-8

    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_tgt, x_src[idx_src[i:e]].T)
        # pred = scores.argmax(axis=0)

        pred = scores.argsort(axis=0)[-top_k:][::-1].T

        for j in range(i, e):
        
            if any(one_top_k in lexicon[idx_src[j]] for one_top_k in pred[j - i]):
                acc += 1.0

            elif do_lemmatized_match and lematized_match(ref=list(lexicon[idx_src[j]]), pred=pred[j - i], w2i_tgt=w2i_tgt, i2w_tgt=i2w_tgt, si_splits=si_splits, si_splits_file_path=si_splits_file_path):
                acc += 1.0
    
    return acc / lexicon_size


def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024, top_k=1, norm=False, norm_src_only=False, \
    do_lemmatized_match=False, w2i_tgt=None, i2w_tgt=None, si_splits=None, si_splits_file_path=None):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)

    idx_src = list(lexicon.keys())

    # if norm_src_only or norm:
    # x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_src)):
        x_src[i] /= np.linalg.norm(x_src[i]) + 1e-8

    # if norm:
    # x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(len(x_tgt)):
        x_tgt[i] /= np.linalg.norm(x_tgt[i]) + 1e-8

    sr = x_src[list(idx_src)]
    sc = np.dot(sr, x_tgt.T)
    similarities = 2 * sc
    sc2 = np.zeros(x_tgt.shape[0])

    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    
    similarities -= sc2[np.newaxis, :]

    # nn = np.argmax(similarities, axis=1).tolist()
    nn = np.argsort(similarities, axis=1).T[-top_k:][::-1].T
    correct = 0.0
    
    for k in range(0, len(lexicon)):
    
        # if nn[k] in lexicon[idx_src[k]]:
        if any(one_top_k in lexicon[idx_src[k]] for one_top_k in nn[k]):
            correct += 1.0

        elif do_lemmatized_match and lematized_match(ref=list(lexicon[idx_src[k]]), pred=nn[k], w2i_tgt=w2i_tgt, i2w_tgt=i2w_tgt, si_splits=si_splits, si_splits_file_path=si_splits_file_path):
            correct += 1.0
    
    return correct / lexicon_size
