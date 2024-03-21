import os
import sys

#lang_pairs = [('en', 'es')]
#lang_pairs = [('es', 'en')]
#lang_pairs = [('en', 'si')]
#lang_pairs = [('si', 'en')]
lang_pairs = [
 # ('es', 'en')]
 # ('de', 'fi'),
 # ('de', 'fr'),
 # ('de', 'hr'),
 # ('de', 'it'),
 # ('de', 'ru'),
 # ('de', 'tr'),
 ('en', 'de'),
 ('de', 'en'),
 # ('en', 'fi'),
 ('en', 'fr'),
 ('fr', 'en'),
 # ('en', 'hr'),
 ('en', 'it'),
 ('it', 'en'),
 ('en', 'ru'),
 ('ru', 'en'),
 ('en', 'tr'),
 ('tr', 'en'),
 ('en', 'zh'),
 ('zh', 'en'),
 ('en', 'ta'),
 ('ta', 'en'),
 ('en', 'ja'),
 ('ja', 'en')
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
]

base = "wiki"
#base = "cc"

max_load = 200_000
data_dir = f"/home/kasunw_22/data"

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
    
    if base == "cc":
        if not os.path.isfile(ROOT_EMB_SRC):
            print(f"[INFO] Downloading {ROOT_EMB_SRC}")
            src_emb_gz = f"{base}.{lang1}.300.vec.gz"
            src_emb_gz_path = os.path.join(data_dir, src_emb_gz)
            os.system(f'wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{src_emb_gz}" -P {data_dir}')
            os.system(f'gzip -dkv {src_emb_gz_path}')
            os.system(f'rm {src_emb_gz_path}')

        if not os.path.isfile(ROOT_EMB_TRG):
            print(f"[INFO] Downloading {ROOT_EMB_TRG}")
            trg_emb_gz = f"{base}.{lang2}.300.vec.gz"
            trg_emb_gz_path = os.path.join(data_dir, trg_emb_gz)
            os.system(f'wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{trg_emb_gz}" -P {data_dir}')
            os.system(f'gzip -dkv {trg_emb_gz_path}')
            os.system(f'rm {trg_emb_gz_path}')

    else: # "wiki"
        if not os.path.isfile(ROOT_EMB_SRC):
            print(f"[INFO] Downloading {ROOT_EMB_SRC}")
            os.system(f'curl -Lo {ROOT_EMB_SRC} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{lang1}.vec')

        if not os.path.isfile(ROOT_EMB_TRG):
            print(f"[INFO] Downloading {ROOT_EMB_TRG}")
            os.system(f'curl -Lo {ROOT_EMB_TRG} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{lang2}.vec')

    if lang1 == "si" or lang2 == "si":
        ROOT_TRAIN_DICT = "{}/{}_{}-{}_trainset_1500_5000_200k.txt".format(data_dir, base, lang1, lang2)
        ROOT_TEST_DICT = "{}/{}_{}-{}_testset_1500_5000_200k.txt".format(data_dir, base, lang1, lang2)
    else:
        ROOT_TRAIN_DICT = "{}/{}-{}.0-5000.txt".format(data_dir, lang1, lang2)
        ROOT_TEST_DICT = "{}/{}-{}.5000-6500.txt".format(data_dir, lang1, lang2)
        
        if not os.path.isfile(ROOT_TRAIN_DICT):
            print(f"[INFO] Downloading {ROOT_TRAIN_DICT}")
            file_name = os.path.basename(ROOT_TRAIN_DICT)
            os.system(f'wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/{file_name}" -P {data_dir}')

        if not os.path.isfile(ROOT_TEST_DICT):
            print(f"[INFO] Downloading {ROOT_TEST_DICT}")
            file_name = os.path.basename(ROOT_TEST_DICT)
            os.system(f'wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/{file_name}" -P {data_dir}')

    SAVE_ROOT = data_dir  # save aligend WEs

    os.system('CUDA_VISIBLE_DEVICES=0  python ./src/main.py --save_aligned_we --l1 {} --l2 {} --self_learning --train_size {} --emb_src_dir {} --tgt_src_dir {} --train_dict_dir {} --test_dict_dir {} --save_dir {} --max_load {} --base {}'.format(lang1, lang2, size_train, ROOT_EMB_SRC, ROOT_EMB_TRG, ROOT_TRAIN_DICT, ROOT_TEST_DICT, SAVE_ROOT, max_load, base))
