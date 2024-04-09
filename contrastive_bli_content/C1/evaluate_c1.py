import io
import os
import sys
import torch
import argparse
import collections
import numpy as np


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def load_lexicon_s2t(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        word_src, word_tgt = word_src.lower(), word_tgt.lower()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))


def load_lexicon_t2s(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_tgt, word_src = line.split()
        word_tgt, word_src = word_tgt.lower(), word_src.lower()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))


# def compute_nn_accuracy_torch(x_src, x_tgt, lexicon, bsz=256, lexicon_size=-1):
#     if lexicon_size < 0:
#         lexicon_size = len(lexicon)
#     idx_src = list(lexicon.keys())
#     acc = 0.0
#     x_src_ = x_src / (torch.norm(x_src, dim=1, keepdim=True) + 1e-9)
#     x_tgt_ = x_tgt / (torch.norm(x_tgt, dim=1, keepdim=True) + 1e-9)
#     for i in range(0, len(idx_src), bsz):
#         e = min(i + bsz, len(idx_src))
#         scores = torch.matmul(x_tgt_, x_src_[idx_src[i:e]].T)
#         pred = torch.argmax(scores,dim=0)
#         pred = pred.cpu().numpy()
#         for j in range(i, e):
#             if pred[j - i] in lexicon[idx_src[j]]:
#                 acc += 1.0
#     return acc / lexicon_size


# def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=256):
#     if lexicon_size < 0:
#         lexicon_size = len(lexicon)
#     idx_src = list(lexicon.keys())
#     x_src_ =x_src /(torch.norm(x_src, dim=1, keepdim=True) + 1e-9)
#     x_tgt_ =x_tgt /(torch.norm(x_tgt, dim=1, keepdim=True) + 1e-9)
#     sr = x_src_[idx_src]
#     sc = torch.zeros(sr.size(0),x_tgt_.size(0)).cuda()
#     for i in range(0, len(idx_src), bsz):
#         e = min(i + bsz, len(idx_src))
#         sc_ = torch.matmul(x_tgt_, sr[i:e].T)
#         sc[i:e] = sc_.T
#     similarities = 2 * sc
#     sc2 = torch.zeros(x_tgt_.size(0)).cuda()
#     for i in range(0, x_tgt_.size(0), bsz):
#         j = min(i + bsz, x_tgt_.size(0))
#         sc_batch = torch.matmul(x_tgt_[i:j,:], x_src_.T)
#         dotprod = torch.topk(sc_batch,k=k,dim=1,sorted=False).values
#         sc2[i:j] = torch.mean(dotprod, dim=1)
#     similarities -= sc2.unsqueeze(0)

#     nn = torch.argmax(similarities, dim=1).cpu().tolist()
#     correct = 0.0
#     for k in range(0, len(lexicon)):
#         if nn[k] in lexicon[idx_src[k]]:
#             correct += 1.0
#     return correct / lexicon_size


def compute_nn_accuracy_torch(x_src, x_tgt, lexicon, bsz=256, lexicon_size=-1, top_k=1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    acc = 0.0
    x_src_ = x_src / (torch.norm(x_src, dim=1, keepdim=True) + 1e-9)
    x_tgt_ = x_tgt / (torch.norm(x_tgt, dim=1, keepdim=True) + 1e-9)
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = torch.matmul(x_tgt_, x_src_[idx_src[i:e]].T)
        # pred = torch.argmax(scores,dim=0)
        pred = scores.argsort(dim=0)[-top_k:].T
        pred = pred.cpu().numpy()
        for j in range(i, e):
            # if pred[j - i] in lexicon[idx_src[j]]:
            if any(one_top_k in lexicon[idx_src[j]] for one_top_k in pred[j - i]):
                acc += 1.0
    return acc / lexicon_size


def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=256, top_k=1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    x_src_ =x_src /(torch.norm(x_src, dim=1, keepdim=True) + 1e-9)
    x_tgt_ =x_tgt /(torch.norm(x_tgt, dim=1, keepdim=True) + 1e-9)
    sr = x_src_[idx_src]
    sc = torch.zeros(sr.size(0),x_tgt_.size(0)).cuda()
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        sc_ = torch.matmul(x_tgt_, sr[i:e].T)
        sc[i:e] = sc_.T
    similarities = 2 * sc
    sc2 = torch.zeros(x_tgt_.size(0)).cuda()
    for i in range(0, x_tgt_.size(0), bsz):
        j = min(i + bsz, x_tgt_.size(0))
        sc_batch = torch.matmul(x_tgt_[i:j,:], x_src_.T)
        dotprod = torch.topk(sc_batch,k=k,dim=1,sorted=False).values
        sc2[i:j] = torch.mean(dotprod, dim=1)
    similarities -= sc2.unsqueeze(0)

    # nn = torch.argmax(similarities, dim=1).cpu().tolist()
    #nn = torch.argsort(similarities, dim=1).T[-top_k:].T.cpu().tolist()  # cannot do argsort in 24GB GPU memory. Not enough memory
    nn = torch.argsort(similarities.cpu(), dim=1).T[-top_k:].T.tolist()
    correct = 0.0
    for k in range(0, len(lexicon)):
        # if nn[k] in lexicon[idx_src[k]]:
        if any(one_top_k in lexicon[idx_src[k]] for one_top_k in nn[k]):
            correct += 1.0
    return correct / lexicon_size


def valid_BLI(train_data_l1, train_data_l2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s, top_k):

    print(f"Top-{top_k} BLI Evaluation...")
    train_data_l1_translation = train_data_l1.cuda()
    train_data_l2_translation = train_data_l2.cuda()
    acc_s2t = compute_nn_accuracy_torch(train_data_l1_translation, train_data_l2_translation, src2tgt, lexicon_size=-1, top_k=top_k) 
    cslsacc_s2t = compute_csls_accuracy(train_data_l1_translation, train_data_l2_translation, src2tgt, lexicon_size=-1, top_k=top_k)  

    acc_t2s = compute_nn_accuracy_torch(train_data_l2_translation, train_data_l1_translation, tgt2src, lexicon_size=-1, top_k=top_k) 
    cslsacc_t2s = compute_csls_accuracy(train_data_l2_translation, train_data_l1_translation, tgt2src, lexicon_size=-1, top_k=top_k)

    BLI_accuracy_l12l2 = (acc_s2t, cslsacc_s2t)
    BLI_accuracy_l22l1 = (acc_t2s, cslsacc_t2s)
    return (BLI_accuracy_l12l2, BLI_accuracy_l22l1) 
 

def not_ASCII(word):
    is_ascii = [ord(x)<256 for x in word]

    if all(is_ascii):
        return False

    return True


prunables = ["si", "zh", "ru", "ta", "ja"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C1 EVALUATION')

    parser.add_argument("--root", type=str, default="./",
                    help="root folder")
    parser.add_argument("--l1", type=str, default=" ",
                    help="l1")
    parser.add_argument("--l2", type=str, default=" ",
                    help="l2")
    # parser.add_argument('--l1_voc', type=str, required=True,
    #                     help='Directory of L1 Vocabulary')
    # parser.add_argument('--l1_emb', type=str, required=True,
    #                     help='Directory of Aligned Static Embeddings for L1')
    # parser.add_argument('--l2_voc', type=str, required=True,
    #                     help='Directory of L2 Vocabulary')
    # parser.add_argument('--l2_emb', type=str, required=True,
    #                     help='Directory of Aligned Static Embeddings for L2')
    parser.add_argument("--train_dict_dir", type=str, default="./",
                    help="train dict directory")
    parser.add_argument("--test_dict_dir", type=str, default="./",
                    help="test dict directory")
    parser.add_argument("--top_k", type=int, default=1,
                    help="top-k retrieval accuracy")
    parser.add_argument('--prune_ascii', action='store_true', 
                    help='Remove ASCII entries from Non-English alphebets')

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    print(f"EVALUATING {args.l1} <--> {args.l2}...")
    sys.stdout.flush()
    l1_voc_path = os.path.join(args.root, f"{args.l1}2{args.l2}_{args.l1}_voc.npy")
    l1_emb_path = os.path.join(args.root, f"{args.l1}2{args.l2}_{args.l1}_emb.pt")
    l2_voc_path = os.path.join(args.root, f"{args.l1}2{args.l2}_{args.l2}_voc.npy")
    l2_emb_path = os.path.join(args.root, f"{args.l1}2{args.l2}_{args.l2}_emb.pt")
    DIR_TEST_DICT = args.test_dict_dir
    DIR_TRAIN_DICT = args.train_dict_dir


    l1_voc = np.load(l1_voc_path, allow_pickle=True).item()
    l2_voc = np.load(l2_voc_path, allow_pickle=True).item()
    l1_emb = torch.load(l1_emb_path)
    l2_emb = torch.load(l2_emb_path)
    
    print(f"[INFO] Original embeddings size {args.l1}: {l1_emb.shape} -- {len(l1_voc)}")
    if args.prune_ascii and args.l1 in prunables:
        print("[INFO] Pruning the vocabulary by removing ASCII entries...")
        words_src = list(l1_voc.keys())
        prune_idx = list(map(not_ASCII, words_src))
        l1_emb = l1_emb[prune_idx]
        # l1_voc = dict(np.array(list(l1_voc.items()))[prune_idx])
        words = np.array(list(l1_voc))[prune_idx]
        l1_voc = {words[i]: i for i in range(len(words))}

        print(f"[INFO] After pruning embeddings size {args.l1}: {l1_emb.shape} -- {len(l1_voc)}")
        
    print(f"[INFO] Original embeddings size {args.l2}: {l2_emb.shape} -- {len(l2_voc)}")
    if args.prune_ascii and args.l2 in prunables:
        print("[INFO] Pruning the vocabulary by removing ASCII entries...")
        words_tgt = list(l2_voc.keys())
        prune_idx = list(map(not_ASCII, words_tgt))
        l2_emb = l2_emb[prune_idx]
        # l2_voc = dict(np.array(list(l2_voc.items()))[prune_idx])
        words = np.array(list(l2_voc))[prune_idx]
        l2_voc = {words[i]: i for i in range(len(words))}

        print(f"[INFO] After pruning embeddings size {args.l2}: {l2_emb.shape} -- {len(l2_voc)}")

    l1_emb = l1_emb / (torch.norm(l1_emb, dim=1, keepdim=True) + 1e-9 )
    l2_emb = l2_emb / (torch.norm(l2_emb, dim=1, keepdim=True) + 1e-9 )

    words_src = list(l1_voc.keys())
    words_tgt = list(l2_voc.keys())

    src2tgt, lexicon_size_s2t = load_lexicon_s2t(DIR_TEST_DICT, words_src, words_tgt)
    tgt2src, lexicon_size_t2s = load_lexicon_t2s(DIR_TEST_DICT, words_tgt, words_src)
    print("lexicon_size_s2t, lexicon_size_t2s", lexicon_size_s2t, lexicon_size_t2s)
    if l1_emb.size(1) < 768:
        accuracy_BLI = valid_BLI(l1_emb, l2_emb, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s, args.top_k)
        print("C1: ", "\nBLI Accuracy L1 to L2 (NN, CSLS): ", accuracy_BLI[0], "\nBLI Accuracy L2 to L1  (NN, CSLS): ", accuracy_BLI[1])
        sys.stdout.flush()
