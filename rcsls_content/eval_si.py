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

parser = argparse.ArgumentParser(description='Evaluation of word alignment')
parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
parser.add_argument("--src_mat", type=str, default='', help="Load source alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--tgt_mat", type=str, default='', help="Load target alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--dico_test", type=str, default='', help="test dictionary")
parser.add_argument("--top_k", type=int, default=10, help="top-k value for accuracy evaluations")
parser.add_argument("--maxload", type=int, default=200000)
parser.add_argument("--nomatch", action='store_true', help="no exact match in lexicon")
params = parser.parse_args()


###### SPECIFIC FUNCTIONS ######
# function specific to evaluation
# the rest of the functions are in utils.py

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

print("========================================")

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
    # x_src = np.dot(x_src, R_src)  # for babylone
    x_src = np.dot(x_src, R_src.T)  # for this repo

src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)

# nnacc = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
# cslsproc = compute_csls_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
nnacc = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, top_k=params.top_k)
cslsproc = compute_csls_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, top_k=params.top_k)
print("NN = %.4f - CSLS = %.4f - Coverage = %.4f" % (nnacc, cslsproc, len(src2tgt) / lexicon_size))
