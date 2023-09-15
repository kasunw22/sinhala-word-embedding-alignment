from ordered_set import OrderedSet

#base="wiki"
base="cc"

src="en"
#src="si"

tgt="si"
#tgt="en"

use_ref_src = True  # use src column of the reference dictionary
swap_our_dict = False  # swap src and tgt of our_dictionary
verify_in_vocabs = False  # check if the our_dictioary words are present in the vocabulary files (False will make the buld faster)
max_same = 5

vocab_base = "/home/kasunw_22/msc/data"

ref_dictionary = f"{vocab_base}/en-de.txt"
our_dictionary = "/home/kasunw_22/msc/sinhala-para-dict/V2/En-Si-dict-FastText-V2.txt"

#out_dictionary = f"{vocab_base}/cc-en-si.txt"
#out_dictionary = f"{vocab_base}/cc-si-en.txt"
#out_dictionary = f"{vocab_base}/wiki-en-si.txt"
out_dictionary = f"{vocab_base}/{base}-{src}-{tgt}.txt"

if base == "cc":
    en_vocab_file = f"{vocab_base}/cc.en.300.vec_words.txt"
    si_vocab_file = f"{vocab_base}/cc.si.300.vec_words.txt"
else:
    en_vocab_file = f"{vocab_base}/wiki.en.vec_words.txt"
    si_vocab_file = f"{vocab_base}/wiki.si.vec_words.txt"

#use_ref_src = True  # use src column of the reference dictionary
#swap_our_dict = False  # swap src and tgt of our_dictionary
#verify_in_vocabs = False  # check if the our_dictioary words are present in the vocabulary files (False will make the buld faster)
#max_same = 5

en_vocab = []

print("Reading English vocabulay...")
with open(en_vocab_file) as f:
    
    for i, line in enumerate(f):

        en_vocab.append(line.strip())


print(f"{len(en_vocab)} words are read...")

si_vocab = []

print("Reading Sinhala vocabulay...")
with open(si_vocab_file) as f:
    
    for i, line in enumerate(f):

        si_vocab.append(line.strip())
        
print(f"{len(si_vocab)} words are read...")


def read_dictionary(file):
    o_src_2_tgt_dict = {}
    o_tgt_2_src_dict = {}
    unique_keys = OrderedSet()
    unique_vals = OrderedSet()

    with open(file) as f:
        
        for line in f:
            src, tgt = line.strip().split()

            # if not swap_src_tgt:
            if not src in o_src_2_tgt_dict:
                o_src_2_tgt_dict[src] = []
            
            o_src_2_tgt_dict[src].append(tgt)

            if not tgt in o_tgt_2_src_dict:
                o_tgt_2_src_dict[tgt] = []
            
            o_tgt_2_src_dict[tgt].append(src)

            unique_keys.add(src)
            unique_vals.add(tgt)
            # else:
            #     if not tgt in o_dict:
            #         o_dict[tgt] = []
                
            #     o_dict[tgt].append(src)
            #     unique_keys.add(tgt)
            #     unique_vals.add(src)
            
    
    return o_src_2_tgt_dict, o_tgt_2_src_dict, unique_keys, unique_vals


def get_unique_src_tgt(file):
    src = OrderedSet()
    tgt = OrderedSet()

    with open(file) as f:
        for i, line in enumerate(f):
            src_w, tgt_w = line.strip().split()
            src.add(src_w)
            tgt.add(tgt_w)

    print(f"src: words={i+1}, unique: {len(src)}")
    print(f"tgt: words={i+1}, unique: {len(tgt)}")

    return src, tgt


def build_alignment_dataset(our_src_2_tgt_dict, our_tgt_2_src_dict, our_keys, our_vals, ref_words, out_file, max_same, src_vocab, tgt_vocab, swap_src_tgt=False, verify_in_vocabs=True):
    print(f"Processing {len(ref_words)} unique words...")
    
    used_keys = set()
    with open(out_file, "w") as f: 
        for i, w in enumerate(ref_words):
            if i % 1000 == 0:
                print(f"[INFO] Processing {i}th word")

            # if i > 100:
            #     break

            if not swap_src_tgt:
                if w in our_keys:
                    n = 0
                    for tgt_w in our_src_2_tgt_dict[w]:
                        if not verify_in_vocabs or (w in src_vocab and tgt_w in tgt_vocab):  # comment this line and unindent the content if you are sure that all the words are present in the vocabularies
                            f.write(f"{w} {tgt_w}\n")
                            n += 1
                        ############################ indent or unindent according to above if-clause

                        if n >= max_same:
                            break
            else:
                if w in our_keys:
                    for src_w in our_src_2_tgt_dict[w]:
                        if src_w in used_keys:
                            continue
                        n = 0
                        for tgt_w in our_tgt_2_src_dict[src_w]:
                            if not verify_in_vocabs or (src_w in src_vocab and tgt_w in tgt_vocab):  # comment this line and unindent the content if you are sure that all the words are present in the vocabularies
                                f.write(f"{src_w} {tgt_w}\n")
                                n += 1
                                used_keys.add(src_w)
                            ############################ indent or unindent according to above if-clause

                            if n >= max_same:
                                break

    print("Done!")


our_src_2_tgt_dict, our_tgt_2_src_dict, our_keys, our_vals = read_dictionary(our_dictionary)
ref_src, ref_tgt = get_unique_src_tgt(ref_dictionary)

ref_words = ref_src if use_ref_src else ref_tgt
# our_words = our_keys if use_src else our_vals
src_vocab = si_vocab if swap_our_dict else en_vocab
tgt_vocab = en_vocab if swap_our_dict else si_vocab

# build_alignment_dataset(our_dict, our_keys, ref_words, out_dictionary, max_same, en_vocab, si_vocab)
build_alignment_dataset(our_src_2_tgt_dict, our_tgt_2_src_dict, our_keys, our_vals, ref_words, out_dictionary, max_same, src_vocab, tgt_vocab, swap_our_dict, verify_in_vocabs)

print(f"Done {out_dictionary}...")
