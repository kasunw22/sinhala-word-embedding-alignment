import os
import sys
# lang_pairs = [('en', 'es')]
lang_pairs = [('en', 'si')]
 # ('de', 'fi'),
 # ('de', 'fr'),
 # ('de', 'hr'),
 # ('de', 'it'),
 # ('de', 'ru'),
 # ('de', 'tr'),
 # ('en', 'de'),
 # ('en', 'fi'),
 # ('en', 'fr'),
 # ('en', 'hr'),
 # ('en', 'it'),
 # ('en', 'ru'),
 # ('en', 'tr'),
 # ('fi', 'fr'),
 # ('fi', 'hr'),
 # ('fi', 'it'),
 # ('fi', 'ru'),
 # ('hr', 'fr'),
 # ('hr', 'it'),
 # ('hr', 'ru'),
 # ('it', 'fr'),
 # ('ru', 'fr'),
 # ('ru', 'it'),
 # ('tr', 'fi'),
 # ('tr', 'fr'),
 # ('tr', 'hr'),
 # ('tr', 'it'),
 # ('tr', 'ru')
# ]

base = "wiki"
# base = "cc"

data_dir = "/home/kasun/Documents/MSc/Research/codes/Fasttext/fastText-master-kasun/alignment/data"

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()

    size_train = "5k" # or "1k"

    if base == "wiki":
        ROOT_EMB_SRC = "{}/{}.{}.vec".format(data_dir, base, lang1)
        ROOT_EMB_TRG = "{}/{}.{}.vec".format(data_dir, base, lang2)
    elif base == "cc":
        ROOT_EMB_SRC = "{}/{}.{}.300.vec".format(data_dir, base, lang1)
        ROOT_EMB_TRG = "{}/{}.{}.300.vec".format(data_dir, base, lang2)
    else:
        raise Exception("base should be either 'wiki' or 'cc'")

    if lang1 == "si" or lang2 == "si":
        ROOT_TEST_DICT = "{}/{}_{}-{}_trainset_1500_5000_200k.txt".format(data_dir, base, lang1, lang2)
        ROOT_TRAIN_DICT = "{}/{}_{}-{}_testset_1500_5000_200k.txt".format(data_dir, base, lang1, lang2)
    else:
        ROOT_TEST_DICT = "{}/{}-{}.0-5000.txt".format(data_dir, lang1, lang2)
        ROOT_TRAIN_DICT = "{}/{}-{}.5000-6500.txt".format(data_dir, lang1, lang2)

    SAVE_ROOT = data_dir  # save aligend WEs

    os.system('CUDA_VISIBLE_DEVICES=0  python ./src/main.py --l1 {} --l2 {} --self_learning --train_size {} --emb_src_dir {} --tgt_src_dir {} --train_dict_dir {} --test_dict_dir {} --save_dir {}'.format(lang1, lang2, size_train, ROOT_EMB_SRC, ROOT_EMB_TRG, ROOT_TRAIN_DICT, ROOT_TEST_DICT, SAVE_ROOT))