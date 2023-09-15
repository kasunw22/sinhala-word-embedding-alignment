import random
import numpy as np


random.seed(42)

# shuffle before train-test split
shuffle_before_split = False

vocab_base = "/home/kasunw_22/msc/data"
#en_vocab_file = f"{vocab_base}/cc.en.300.vec_words.txt"
#si_vocab_file = f"{vocab_base}/cc.si.300.vec_words.txt"
en_vocab_file = f"{vocab_base}/wiki.en.vec_words.txt"
si_vocab_file = f"{vocab_base}/wiki.si.vec_words.txt"

# base_dictionary = "/home/kasunw_22/msc/sinhala-para-dict/V2/En-Si-dict-FastText-V2.txt"
base_dictionary = "/home/kasunw_22/msc/data/en-si.txt"

maxload = 200000

train_size = 5000
test_size = 1500

#train_size = 10000
#test_size = 3000

assert train_size >= test_size

# trainset_name = f"{vocab_base}/wiki_maximum_trainset_{train_size}_200k.txt"
# testset_name = f"{vocab_base}/wiki_maximum_testset_{test_size}_200k.txt"

trainset_name = f"{vocab_base}/wiki_en-si_trainset_{train_size}_200k-2.txt"
testset_name = f"{vocab_base}/wiki_en-si_testset_{test_size}_200k-2.txt"

#trainset_name = f"{vocab_base}/cc_en-si_trainset_{train_size}_200k.txt"
#testset_name = f"{vocab_base}/cc_en-si_testset_{test_size}_200k.txt"

max_vocab_size = maxload  #int(maxload * 1.25)
train_percent = train_size / (train_size + test_size)

# best_pairs = []

en_vocab = []

print("Reading English vocabulay...")
with open(en_vocab_file) as f:
    
    for i, line in enumerate(f):

        if i < max_vocab_size:
            en_vocab.append(line.strip())
        else:
            break


print(f"{len(en_vocab)} words are read...")

si_vocab = []

print("Reading Sinhala vocabulay...")
with open(si_vocab_file) as f:
    
    for i, line in enumerate(f):

        if i < max_vocab_size:
            si_vocab.append(line.strip())
        else:
            break
        
print(f"{len(si_vocab)} words are read...")

print(f"Reading en-si dictionary..")

base_dict = {}

print(f"Reading the base dictionary..")
with open(base_dictionary) as f:
    
    for line in f:
        src, tgt = line.strip().split()
        
        if src not in base_dict:
            base_dict[src] = []

        base_dict[src].append(tgt)

# with open(base_dictionary) as f:
#     dictionary = f.readlines()

# print(f"Extracting available pairs..")
# for i, line in enumerate(dictionary):
#     en_w, si_w = line.strip().split()

#     if en_w in en_vocab and si_w in si_vocab:
#         best_pairs.append(f"{en_w} {si_w}\n")

all_src_words = list(base_dict.keys())

if shuffle_before_split:
    random.shuffle(all_src_words)

best_pairs_len = len(all_src_words)
cut_off_idx = int(best_pairs_len * train_percent)


def find_idxs(all_words, size, ignore_src_words=[]):
    # unique_ids = []
    unique_w = []
    unique_end = 0
    
    for idx, sr_w in enumerate(all_words):
        # sr_w, tg_w = wp.strip().split()
        # sr_w = wp
        if sr_w not in unique_w + ignore_src_words:
            unique_w.append(sr_w)
            # unique_ids.append(idx)
        if len(unique_w)>= size:
            unique_end = idx + 1
            break

    # return np.array(unique_ids), unique_w, unique_end
    return unique_w, unique_end

# def find_end_idx(pairs, size, ignore_src_words=[]):
#     unique_w = []
#     unique_ids = []
#     unique_end = 0
#     unique_start = 0
    
#     for idx, wp in enumerate(pairs):
#         sr_w, tg_w = wp.strip().split()
#         if sr_w not in unique_w:
#             unique_w.append(sr_w)
#         if len(unique_w)>= size:
#             unique_end = idx + 1
#             break

#     return unique_start, unique_end


# train_end = find_end_idx(best_pairs[: cut_off_idx], train_size)
# test_end = find_end_idx(best_pairs[cut_off_idx:], test_size)

# train_set = best_pairs[: cut_off_idx][:train_end]
# test_set = best_pairs[cut_off_idx:][:test_end]

all_src_words = np.array(all_src_words)

print("Finding train indexes...")
train_words, end_idx = find_idxs(all_src_words, train_size)
# train_ids, train_words, end_idx = find_idxs(best_pairs, train_size)

print("Finding test indexes...")
test_words, _ = find_idxs(all_src_words[end_idx:], test_size, train_words)
# test_ids, test_words, _ = find_idxs(best_pairs[end_idx:], test_size, train_words)


def build_dictionary(base_dict, words, max_same=5):
    dataset = []
    for w in words:
        for tgt_w in base_dict[w][:max_same]:
            dataset.append(f"{w} {tgt_w}\n")

    return dataset

# train_set = best_pairs[train_ids]
# test_set = best_pairs[end_idx:][test_ids]

train_set = build_dictionary(base_dict, train_words)
test_set = build_dictionary(base_dict, test_words)

print(f"Writing trainset..")
with open(trainset_name, "w") as f:
    f.write("".join(train_set))

print(f"Writing testset..")
with open(testset_name, "w") as f:
    f.write("".join(test_set))

print("Done!")

