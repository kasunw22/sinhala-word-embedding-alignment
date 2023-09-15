import csv
import pandas as pd
from string import punctuation

# import nltk
# from nltk.corpus import stopwords

import spacy
en = spacy.load('en_core_web_sm')

en_stops = en.Defaults.stop_words

# en_stops = list(set(stopwords.words('english')))
# print(en_stops)

si_sw_file = "/home/kasunw_22/msc/data/si-stop-words.txt"

with open(si_sw_file, encoding='utf-8') as f:
    si_stops = f.readlines()

si_stops = [w.strip() for w in si_stops]

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


def dict_coverage(dictionary, src_vocab, tgt_vocab):
    count = 0

    for src, tgt in dictionary:

        if (src in src_vocab) and (tgt in tgt_vocab):
            count += 1
    
    print(f"Dictionary > total entries: {len(dictionary)}, coverage count: {count}")
    return (count / len(dictionary))




swap_cols = False

vocab_base = "/home/kasunw_22/msc/data"

if use_wiki:
    en_vocab_file = f"{vocab_base}/wiki.en.vec_words.txt"
    si_vocab_file = f"{vocab_base}/wiki.si.vec_words.txt"
else:
    en_vocab_file = f"{vocab_base}/cc.en.300.vec_words.txt"
    si_vocab_file = f"{vocab_base}/cc.si.300.vec_words.txt"

# reading the dictionary
dataset_name = source_file.rsplit("/", 1)[-1]

print(f"Reading the Dataset {dataset_name}...")

if source_file.rsplit(".", 1)[-1] == "tsv":
    sep="\t"
elif source_file.rsplit(".", 1)[-1] == "csv":
    sep = ","
else:
    sep = " "

df = pd.read_csv(source_file, header=None, sep=sep)

if swap_cols:
    columns={0:"si", 1: "en"}
    df.rename(columns=columns, inplace=True)
    df = df[["en", "si"]]
else:
    columns={0:"en", 1: "si"}
    df.rename(columns=columns, inplace=True)


print(df)
# print(df.values.tolist())

dictionary = df.values.tolist()

# print(dictionary)

# exit(0)

unique_en = df.en.unique().tolist()
unique_si = df.si.unique().tolist()

df_wo_en_stop = df[~df.en.isin(en_stops)]
unique_en_wo_sw = df_wo_en_stop.en.unique().tolist()

df_wo_si_stop = df[~df.si.isin(si_stops)]
unique_si_wo_sw = df_wo_si_stop.si.unique().tolist()

# reading the FastText words
en_vocab = []
en_vocab_wo_sw = []

print("Reading cc English vocabulay...")
with open(en_vocab_file) as f:
    for line in f:
        entry = line.strip()

        if not is_punc(entry):
            en_vocab.append(entry)

            if entry not in en_stops:
                en_vocab_wo_sw.append(entry)

print(f"{len(en_vocab)} words are read...")

si_vocab = []
si_vocab_wo_sw = []
print("Reading cc Sinhala vocabulay...")
with open(si_vocab_file) as f:
    for line in f:
        entry = line.strip()

        # if (not is_ASCII(entry) and (not is_punc(entry))):
        if not is_punc(entry):
            si_vocab.append(entry)

            if entry not in si_stops:
                si_vocab_wo_sw.append(entry)

print(f"{len(si_vocab)} words are read...")

###################################################
en_vocab_wiki = []
en_vocab_wo_sw_wiki = []

print("Reading wiki English vocabulay...")
with open(en_vocab_file_wiki) as f:
    for line in f:
        entry = line.strip()

        if not is_punc(entry):
            en_vocab_wiki.append(entry)

            if entry not in en_stops:
                en_vocab_wo_sw_wiki.append(entry)

print(f"{len(en_vocab_wiki)} words are read...")

si_vocab_wiki = []
si_vocab_wo_sw_wiki = []
print("Reading wiki Sinhala vocabulay...")
with open(si_vocab_file_wiki) as f:
    for line in f:
        entry = line.strip()

        # if (not is_ASCII(entry) and (not is_punc(entry))):
        if not is_punc(entry):
            si_vocab_wiki.append(entry)

            if entry not in si_stops:
                si_vocab_wo_sw_wiki.append(entry)

print(f"{len(si_vocab_wiki)} words are read...")

print("\nFastText stats:=======================")
print(f"Total English words (cc): {len(en_vocab)}")
print(f"Total Sinhala words (cc): {len(si_vocab)}")

print(f"Total English words (wiki): {len(en_vocab_wiki)}")
print(f"Total Sinhala words (wiki): {len(si_vocab_wiki)}")

print("\nDictionary Stats:=======================")
print(f"Total entries: {len(df)}")
print(f"Total English words w/o Stopwords: {len(df_wo_en_stop)}\n")
print(f"Total Sinhala words w/o Stopwords: {len(df_wo_si_stop)}\n")

print(f"Unique English words: {len(unique_en)}")
print(f"Unique English words %: {len(unique_en)*100/len(df)}")
# print(f"Unique English word coverage (WRONG): {len(unique_en)*100/len(en_vocab)} %")
print(f"Unique English words (w/o stopwords): {len(unique_en_wo_sw)}")
print(f"Unique English words % (w/o stopwords): {len(unique_en_wo_sw)*100/len(df_wo_en_stop)}")

print(f"Unique Sinhala words: {len(unique_si)}")
print(f"Unique Sinhala words %: {len(unique_si)*100/len(df)}")
print(f"Unique Sinhala words (w/o stopwords): {len(unique_si_wo_sw)}")
print(f"Unique Sinhala words % (w/o stopwords): {len(unique_si_wo_sw)*100/len(df_wo_si_stop)}")

print("\nStats w.r.t. cc model...............")
# print(f"Unique English word coverage (w/o stopwords): {len(unique_en_wo_sw)*100/len(en_vocab_wo_sw)} %")
print(f"English look-up precision: {lookup_precision(en_vocab, unique_en)*100} %")
# print(f"Unique Sinhala words coverage (WRONG): {len(unique_si)*100/len(si_vocab)} %")
print(f"Sinhala look-up precision: {lookup_precision(si_vocab, unique_si)*100} %")

print("\nStats w.r.t. wiki model...............")
# print(f"Unique English word coverage (w/o stopwords): {len(unique_en_wo_sw)*100/len(en_vocab_wo_sw)} %")
print(f"English look-up precision: {lookup_precision(en_vocab_wiki, unique_en)*100} %")
# print(f"Unique Sinhala words coverage (WRONG): {len(unique_si)*100/len(si_vocab)} %")
print(f"Sinhala look-up precision: {lookup_precision(si_vocab_wiki, unique_si)*100} %")
print()

print(f"Dictionary coverage (cc): {dict_coverage(dictionary, en_vocab, si_vocab)*100}%")
print(f"Dictionary coverage (wiki): {dict_coverage(dictionary, en_vocab_wiki, si_vocab_wiki)*100}%")
