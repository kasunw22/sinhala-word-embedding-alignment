import random
import numpy as np


random.seed(42)

# shuffle before train-test split
shuffle_before_split = False

#base="cc"
base="wiki"

src="si"
#src="en"

tgt="en"
#tgt="si"

vocab_base = "/home/kasunw_22/msc/data"

if base == "wiki":
    en_vocab_file = f"{vocab_base}/wiki.en.vec_words.txt"
    si_vocab_file = f"{vocab_base}/wiki.si.vec_words.txt"
else:
    en_vocab_file = f"{vocab_base}/cc.en.300.vec_words.txt"
    si_vocab_file = f"{vocab_base}/cc.si.300.vec_words.txt"

#base_dictionary = "/home/kasunw_22/msc/sinhala-para-dict/V2/En-Si-dict-FastText-V2.txt"
base_dictionary = f"/home/kasunw_22/msc/data/{base}-{src}-{tgt}.txt"

maxload = 200000

train_size = 5000
test_size = 1500

#train_size = 10000
#test_size = 3000

assert train_size >= test_size

trainset_name = f"{vocab_base}/{base}_{src}-{tgt}_trainset_{test_size}_{train_size}_200k.txt"
testset_name = f"{vocab_base}/{base}_{src}-{tgt}_testset_{test_size}_{train_size}_200k.txt"

max_vocab_size = maxload  #int(maxload * 1.25)
train_percent = train_size / (train_size + test_size)

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

if src == "en":
    src_vocab = en_vocab
    tgt_vocab = si_vocab
else:
    src_vocab = si_vocab
    tgt_vocab = en_vocab


def read_dictionary(file_path, src_vocab, tgt_vocab):
    base_dict = {}

    print(f"Reading the dictionary..")
    with open(file_path) as f:
        
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print(f"Reading {i}th entry")

            #if i >= 10000:
            #    break

            src, tgt = line.strip().split()
            
            if (src not in src_vocab) or (tgt not in tgt_vocab):
                continue

            if src not in base_dict:
                base_dict[src] = []

            base_dict[src].append(tgt)
        
    return base_dict


base_dict = read_dictionary(base_dictionary, src_vocab, tgt_vocab)

#print(base_dict)
#exit(0)

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


print("Building trainset...")
train_set = build_dictionary(base_dict, train_words, src_vocab, tgt_vocab)

print("Building testset...")
test_set = build_dictionary(base_dict, test_words, src_vocab, tgt_vocab)

print(f"Writing trainset..")
with open(trainset_name, "w") as f:
    f.write("".join(train_set))
print(f"{trainset_name} is done!")

print(f"Writing testset..")
with open(testset_name, "w") as f:
    f.write("".join(test_set))
print(f"{testset_name} is done!")

print("Done!")
