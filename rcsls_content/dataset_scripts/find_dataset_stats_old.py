import csv
import pandas as pd
from string import punctuation

# import nltk
# from nltk.corpus import stopwords

import spacy
en = spacy.load('en_core_web_sm')

en_stops = en.Defaults.stop_words

# en_stops = list(set(stopwords.words('english')))
print(en_stops)


def is_ASCII(word):
    is_ascii = [ord(x)<256 for x in word]
    
    if any(is_ascii):
        return True
    
    return False

def is_punc(word):
    # print(word)
    # print([x for x in word])
    is_punc = [(x in punctuation) for x in word]
    
    if any(is_punc):
        return True
    
    return False

def lookup_precision(ref_list, tgt_list):
    count = 0

    for tw in tgt_list:

        if tw in ref_list:
            count += 1
    
    return (count / len(tgt_list))


source_file = "/home/kasunw_22/msc/sinhala-para-dict/V2/En-Si-dict-FastText-V2.tsv"
#source_file = "../data/wiki_maximum_trainset_5000_200k.txt"
#source_file = "../data/wiki_maximum_testset_1500_200k.txt"

separator = "\t"
use_wiki = False

vocab_base = "/home/kasunw_22/msc/data"

if use_wiki:
    en_vocab_file = f"{vocab_base}/wiki.en.vec_words.txt"
    si_vocab_file = f"{vocab_base}/wiki.si.vec_words.txt"
else:
    en_vocab_file = f"{vocab_base}/cc.en.300.vec_words.txt"
    si_vocab_file = f"{vocab_base}/cc.si.300.vec_words.txt"

# reading the dictionary

print("Reading the Dataset...")
df = pd.read_csv(source_file, header=None, sep=separator)
df.rename(columns={0:"en", 1: "si"}, inplace=True)
print(df)

unique_en = df.en.unique().tolist()
unique_si = df.si.unique().tolist()

df_wo_en_stop = df[~df.en.isin(en_stops)]
unique_en_wo_sw = df_wo_en_stop.en.unique().tolist()

# reading the FastText words
en_vocab = []
en_vocab_wo_sw = []

print("Reading English vocabulay...")
with open(en_vocab_file) as f:
    for line in f:
        entry = line.strip()

        if not is_punc(entry):
            en_vocab.append(entry)

            if entry not in en_stops:
                en_vocab_wo_sw.append(entry)

print(f"{len(en_vocab)} words are read...")

si_vocab = []

print("Reading Sinhala vocabulay...")
with open(si_vocab_file) as f:
    for line in f:
        entry = line.strip()

        if (not is_ASCII(entry) and (not is_punc(entry))):
            si_vocab.append(entry)

print(f"{len(si_vocab)} words are read...")

print("\nFastText stats:=======================")
print(f"Total English words: {len(en_vocab)}")
print(f"Total Sinhala words: {len(si_vocab)}")

print("\nDictionary Stats:=======================")
print(f"Total entries: {len(df)}")
print(f"Total entries w/o English Stopwords: {len(df_wo_en_stop)}")

print(f"Unique English words: {len(unique_en)}")
print(f"Unique English words %: {len(unique_en)*100/len(df)}")
# print(f"Unique English word coverage (WRONG): {len(unique_en)*100/len(en_vocab)} %")
print(f"Unique English words (w/o stopwords): {len(unique_en_wo_sw)}")
print(f"Unique English words % (w/o stopwords): {len(unique_en_wo_sw)*100/len(df_wo_en_stop)}")

# print(f"Unique English word coverage (w/o stopwords): {len(unique_en_wo_sw)*100/len(en_vocab_wo_sw)} %")
print(f"English look-up precision: {lookup_precision(en_vocab, unique_en)*100} %")
print()
print(f"Unique Sinhala words: {len(unique_si)}")
print(f"Unique Sinhala words %: {len(unique_si)*100/len(df)}")
# print(f"Unique Sinhala words coverage (WRONG): {len(unique_si)*100/len(si_vocab)} %")
print(f"Sinhala look-up precision: {lookup_precision(si_vocab, unique_si)*100} %")

