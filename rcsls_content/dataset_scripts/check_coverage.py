vocab_base = "/home/kasunw_22/msc/data"
en_vocab_file = f"{vocab_base}/cc.en.300.vec_words.txt"
si_vocab_file = f"{vocab_base}/cc.si.300.vec_words.txt"

#dictionary_files = [f"{vocab_base}/en-si-1500-v2-dict-random.txt", f"{vocab_base}/en-si-5000-v2-dict-random.txt"]
dictionary_files = [f"{vocab_base}/maximum_testset_1500_200k.txt", f"{vocab_base}/maximum_trainset_5000_200k.txt"]

en_vocab = []

print("Reading English vocabulay...")
with open(en_vocab_file) as f:
    for line in f:
        en_vocab.append(line.strip())

print(f"{len(en_vocab)} words are read...")

si_vocab = []

print("Reading Sinhala vocabulay...")
with open(si_vocab_file) as f:
    for line in f:
        si_vocab.append(line.strip())

print(f"{len(si_vocab)} words are read...")

for dictionary_file in dictionary_files:

    # dictionary_file = "sorted-en-si-dictionary-single-word-extended.txt"

    print(f"Reading en-si dictionary {dictionary_file}..")

    with open(dictionary_file) as f:
        dictionary = f.readlines()


    avaible_count = 0

    for i, line in enumerate(dictionary):
        en_w, si_w = line.strip().split()

        if en_w in en_vocab and si_w in si_vocab:
            avaible_count += 1

    print(f"No of entries: {(i + 1)}, Availabe entries in the vocabulary: {avaible_count}")
    print(f"Coverage: {avaible_count / (i + 1)}")
print("Done!")
