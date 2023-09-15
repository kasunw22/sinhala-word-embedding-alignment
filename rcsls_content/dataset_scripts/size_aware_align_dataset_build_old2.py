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

#base_dictionary = "/home/kasunw_22/msc/sinhala-para-dict/V2/En-Si-dict-FastText-V2.txt"
#base_dictionary = "/home/kasunw_22/msc/data/cc-en-si.txt"
base_dictionary = "/home/kasunw_22/msc/data/wiki-en-si.txt"

maxload = 200000

train_size = 5000
test_size = 1500

#train_size = 10000
#test_size = 3000

assert train_size >= test_size

trainset_name = f"{vocab_base}/wiki_en-si_trainset_{test_size}_{train_size}_200k.txt"
testset_name = f"{vocab_base}/wiki_en-si_testset_{test_size}_{train_size}_200k.txt"

#trainset_name = f"{vocab_base}/cc_en-si_trainset_{test_size}_{train_size}_200k.txt"
#testset_name = f"{vocab_base}/cc_en-si_testset_{test_size}_{train_size}_200k.txt"

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


def read_dictionary(file_path, src_vocab, tgt_vocab):
    base_dict = {}

    print(f"Reading the dictionary..")
    with open(file_path) as f:
        
        for line in f:
            src, tgt = line.strip().split()
            
            if (src not in src_vocab) or (tgt not in tgt_vocab):
                continue

            if src not in base_dict:
                base_dict[src] = []

            base_dict[src].append(tgt)
        
    return base_dict


base_dict = read_dictionary(base_dictionary, en_vocab, si_vocab)

all_src_words = list(base_dict.keys())

if shuffle_before_split:
    random.shuffle(all_src_words)

best_pairs_len = len(all_src_words)
cut_off_idx = int(best_pairs_len * train_percent)


def find_idxs(all_words, size, ignore_src_words=[]):
    unique_w = []
    unique_end = 0
    
    for idx, sr_w in enumerate(all_words):

        if sr_w not in unique_w + ignore_src_words:
            unique_w.append(sr_w)

        if len(unique_w)>= size:
            unique_end = idx + 1
            break

    return unique_w, unique_end


all_src_words = np.array(all_src_words)

print("Finding train indexes...")
train_words, end_idx = find_idxs(all_src_words, train_size)

print("Finding test indexes...")
test_words, _ = find_idxs(all_src_words[end_idx:], test_size, train_words)

def build_dictionary(base_dict, words, src_vocab, tgt_vocab, max_same=5):
    dataset = []
    for w in words:
        # for tgt_w in base_dict[w][:max_same]:
        n = 0
        for tgt_w in base_dict[w]:
            if w in src_vocab and tgt_w in tgt_vocab:
                dataset.append(f"{w} {tgt_w}\n")
                n += 1

            if n >= max_same:
                break

    return dataset


train_set = build_dictionary(base_dict, train_words, en_vocab, si_vocab)
test_set = build_dictionary(base_dict, test_words, en_vocab, si_vocab)
'''
def build_dictionary(base_dict, words, max_same=5):
    dataset = []
    for w in words:
        for tgt_w in base_dict[w][:max_same]:
            dataset.append(f"{w} {tgt_w}\n")

    return dataset


train_set = build_dictionary(base_dict, train_words)
test_set = build_dictionary(base_dict, test_words)
'''
print(f"Writing trainset..")
with open(trainset_name, "w") as f:
    f.write("".join(train_set))

print(f"Writing testset..")
with open(testset_name, "w") as f:
    f.write("".join(test_set))

print("Done!")

