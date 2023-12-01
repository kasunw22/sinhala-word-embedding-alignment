from utils_si import *


def extract_words(fname, out_fo, start=0, last=None, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    if last is not None:
        n = min(n, last)

    words = []
    for i, line in enumerate(fin):
        if start > i:
            continue

        if i >= n:
            break
        
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        out_fo.write(f"{word}\n")

        if verbose and (i+1)%10000==0:
            print(f"{i+1} Done")
    
    if verbose:
        print("Writing done")


#model_name = "data/cc.si.300.vec"
model_name = "data/cc.en.300.vec"
#model_name = "data/wiki.en.vec"
#model_name = "data/wiki.si.vec"

f = open(f"{model_name}_words.txt", "w")

extract_words(model_name, f)
