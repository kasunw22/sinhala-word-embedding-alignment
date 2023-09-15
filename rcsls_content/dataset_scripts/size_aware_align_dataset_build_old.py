import random


random.seed(42)

vocab_base = "/home/kasunw_22/msc/data"
#en_vocab_file = f"{vocab_base}/cc.en.300.vec_words.txt"
#si_vocab_file = f"{vocab_base}/cc.si.300.vec_words.txt"
en_vocab_file = f"{vocab_base}/wiki.en.vec_words.txt"
si_vocab_file = f"{vocab_base}/wiki.si.vec_words.txt"

base_dictionary = "/home/kasunw_22/msc/sinhala-para-dict/V2/En-Si-dict-FastText-V2.txt"

maxload = 200000

train_size = 5000
test_size = 1500

#train_size = 10000
#test_size = 3000

assert train_size >= test_size

trainset_name = f"{vocab_base}/wiki_maximum_trainset_{train_size}_200k.txt"
testset_name = f"{vocab_base}/wiki_maximum_testset_{test_size}_200k.txt"

max_vocab_size = maxload  #int(maxload * 1.25)
train_percent = train_size / (train_size + test_size)

best_pairs = []

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

with open(base_dictionary) as f:
    dictionary = f.readlines()

print(f"Extracting available pairs..")
for i, line in enumerate(dictionary):
    en_w, si_w = line.strip().split()

    if en_w in en_vocab and si_w in si_vocab:
        best_pairs.append(f"{en_w} {si_w}\n")

random.shuffle(best_pairs)

best_pairs_len = len(best_pairs)
cut_off_idx = int(best_pairs_len * train_percent)



def find_end_idx(pairs, size):
    unique_w = []
    unique_end = 0
    
    for idx, wp in enumerate(pairs):
        sr_w, tg_w = wp.strip().split()
        if sr_w not in unique_w:
            unique_w.append(sr_w)
        if len(unique_w)>= size:
            unique_end = idx + 1
            break

    return unique_end


train_end = find_end_idx(best_pairs[: cut_off_idx], train_size)
test_end = find_end_idx(best_pairs[cut_off_idx:], test_size)

train_set = best_pairs[: cut_off_idx][:train_end]
test_set = best_pairs[cut_off_idx:][:test_end]

print(f"Writing trainset..")
with open(trainset_name, "w") as f:
    f.write("".join(train_set))

print(f"Writing testset..")
with open(testset_name, "w") as f:
    f.write("".join(test_set))

print("Done!")
