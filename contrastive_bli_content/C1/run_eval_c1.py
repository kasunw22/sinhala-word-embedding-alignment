import os
import sys

#lang_pairs = [('en', 'es')]
#lang_pairs = [('es', 'en')]
#lang_pairs = [('en', 'fr'), ('fr', 'en')]
#lang_pairs = [('en', 'it'), ('it', 'en')]
#lang_pairs = [('en', 'ru'), ('ru', 'en')]
#lang_pairs = [('en', 'de'), ('de', 'en')]
#lang_pairs = [('en', 'ta'), ('ta', 'en')]
#lang_pairs = [('en', 'zh'), ('zh', 'en')]
lang_pairs = [('en', 'tr'), ('tr', 'en')]
#lang_pairs = [('en', 'ja'), ('ja', 'en')]
#lang_pairs = [('fr', 'en')]
#lang_pairs = [('en', 'si')]
#lang_pairs = [('si', 'en')]
# lang_pairs = [('de', 'fi'),
#  ('de', 'fr'),
#  ('de', 'hr'),
#  ('de', 'it'),
#  ('de', 'ru'),
#  ('de', 'tr'),
#  ('en', 'de'),
#  ('en', 'fi'),
#  ('en', 'fr'),
#  ('en', 'hr'),
#  ('en', 'it'),
#  ('en', 'ru'),
#  ('en', 'tr'),
#  ('fi', 'fr'),
#  ('fi', 'hr'),
#  ('fi', 'it'),
#  ('fi', 'ru'),
#  ('hr', 'fr'),
#  ('hr', 'it'),
#  ('hr', 'ru'),
#  ('it', 'fr'),
#  ('ru', 'fr'),
#  ('ru', 'it'),
#  ('tr', 'fi'),
#  ('tr', 'fr'),
#  ('tr', 'hr'),
#  ('tr', 'it'),
#  ('tr', 'ru')]

base = "wiki"
#base = "cc"

#ASCII_PRUNE_VOCAB = True
ASCII_PRUNE_VOCAB = False

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()

    ROOT_DIR = f"/home/kasunw_22/data/{base}/5k"
    DATA_DIR = "/home/kasunw_22/data"
    
    if lang1 == "si" or lang2 == "si":
        TRAIN_DICT = os.path.join(DATA_DIR, f"/{base}_{lang1}-{lang2}_trainset_1500_5000_200k.txt")
        TEST_DICT = os.path.join(DATA_DIR, f"{base}_{lang1}-{lang2}_testset_1500_5000_200k.txt")
    else:
        TRAIN_DICT = os.path.join(DATA_DIR, f"{lang1}-{lang2}.0-5000.txt")
        TEST_DICT = os.path.join(DATA_DIR, f"{lang1}-{lang2}.5000-6500.txt")

    for top_k in [1, 5, 10]:
        if ASCII_PRUNE_VOCAB:
            os.system('CUDA_VISIBLE_DEVICES=0  python evaluate_c1.py --l1 {} --l2 {} --root {} --train_dict_dir {} --test_dict_dir {} --top_k {} --prune_ascii'.format(lang1, lang2, ROOT_DIR, TRAIN_DICT, TEST_DICT, top_k))
        else:
            os.system('CUDA_VISIBLE_DEVICES=0  python evaluate_c1.py --l1 {} --l2 {} --root {} --train_dict_dir {} --test_dict_dir {} --top_k {}'.format(lang1, lang2, ROOT_DIR, TRAIN_DICT, TEST_DICT, top_k))
