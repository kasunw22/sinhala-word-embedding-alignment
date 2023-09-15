from ordered_set import OrderedSet


# ref_dictionary = "/home/kasun/Downloads/de-en.0-5000.txt"
ref_dictionary = "/home/kasunw_22/msc/data/en-de.txt"
our_dictionary = "/home/kasunw_22/msc/sinhala-para-dict/V2/En-Si-dict-FastText-V2.txt"
out_dictionary = "/home/kasunw_22/msc/data/en-si.txt"

use_src = True
max_same = 5

def read_dictionary(file):
    o_dict = {}
    unique_keys = OrderedSet()

    with open(file) as f:
        
        for line in f:
            src, tgt = line.strip().split()
        
            if not src in o_dict:
                o_dict[src] = []
            
            unique_keys.add(src)
            o_dict[src].append(tgt)
    
    return o_dict, unique_keys


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
    # return list(src), list(tgt)


def build_alignment_dataset(our_dict, our_keys, ref_words, out_file, max_same):
    with open(out_file, "w") as f: 
        for w in ref_words:
            if w in our_keys:
                for tgt_w in our_dict[w][:max_same]:
                    f.write(f"{w} {tgt_w}\n")

    print("Done!")

our_dict, our_keys = read_dictionary(our_dictionary)
ref_src, ref_tgt = get_unique_src_tgt(ref_dictionary)

# print(f"our_keys: {our_keys}\n")
# print("--------------------------------")
# print(f"ref_src: {ref_src[:50]}")
# print(f"ref_src: {[x for x in ref_src]}")

ref_words = ref_src if use_src else ref_tgt
build_alignment_dataset(our_dict, our_keys, ref_words, out_dictionary, max_same)

