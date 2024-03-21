import os
import sys
import time

#lang_pairs = [('en', 'es')]
#lang_pairs = [('es', 'en')]
#lang_pairs = [('en', 'si')]
#lang_pairs = [('si', 'en')]
lang_pairs = [('en', 'fr'), ('fr', 'en')]
#lang_pairs = [('en', 'it'), ('it', 'en')]
#lang_pairs = [('en', 'ru'), ('ru', 'en')]
#lang_pairs = [('en', 'de'), ('de', 'en')]
#lang_pairs = [('en', 'ta'), ('ta', 'en')]
#lang_pairs = [('en', 'zh'), ('zh', 'en')]
#lang_pairs = [('en', 'tr'), ('tr', 'en')]
#lang_pairs = [('en', 'ja'), ('ja', 'en')]
"""
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
]
"""
#Hyper-parameters to reproduce our results.
gpuid = '0,1'
# train_size = '1k' #or '5k'
train_size = '5k' #or '1k'
base = "wiki"
#base = "cc"
# DIR_NEW = "./BLI/"
DIR_NEW = f"/home/kasunw_22/data/{base}/BLI/"
data_dir = "/home/kasunw_22/data"
# output_root = "/media/data/bert_models_final/"
output_root = f"/home/kasunw_22/data/{base}/bert_models_final/"
random_seed = 33
# model_dir = "bert-base-multilingual-uncased" # "google/mt5-small", "xlm-mlm-100-1280"
model_dir = "sentence-transformers/LaBSE" # "bert-base-multilingual-uncased" # "google/mt5-small", "xlm-mlm-100-1280"
epoch = 5 # 6 for "google/mt5-small" 
train_batch_size = 100 # 50 for "xlm-mlm-100-1280"
learning_rate = 2e-5 # 6e-4 for "google/mt5-small" 
max_length = 6
checkpoint_step = 9999999
infoNCE_tau = 0.1
agg_mode = "cls" if "LaBSE" not in model_dir else "pooler"
num_neg = 28
num_iter = 1
template = 0
neg_max = 60000 
lambda_ = 0.2
output_dir_postfix = ""

assert agg_mode in ["cls", "mean_pool", "pooler"]

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()


    # ROOT_FT = "/media/data/SAVE{}/".format(train_size)
    ROOT_FT = "/home/kasunw_22/data/{}/{}/".format(base, train_size)
    l1_voc = ROOT_FT + "{}2{}_{}_voc.npy".format(lang1,lang2,lang1)
    l1_emb = ROOT_FT + "{}2{}_{}_emb.pt".format(lang1,lang2,lang1)
    l2_voc = ROOT_FT + "{}2{}_{}_voc.npy".format(lang1,lang2,lang2)
    l2_emb = ROOT_FT + "{}2{}_{}_emb.pt".format(lang1,lang2,lang2)
    # DIR_TEST_DICT = "/media/data/xling-eval/bli_datasets/{}-{}/yacle.test.freq.2k.{}-{}.tsv".format(lang1,lang2,lang1,lang2)
    # DIR_TRAIN_DICT = "media/data/xling-eval/bli_datasets/{}-{}/yacle.train.freq.{}.{}-{}.tsv".format(lang1,lang2,train_size,lang1,lang2)
    if lang1 == "si" or lang2 == "si":
        DIR_TRAIN_DICT = "{}/{}_{}-{}_trainset_1500_5000_200k.txt".format(data_dir, base, lang1, lang2)
        DIR_TEST_DICT = "{}/{}_{}-{}_testset_1500_5000_200k.txt".format(data_dir, base, lang1, lang2)
    else:
        DIR_TRAIN_DICT = "{}/{}-{}.0-5000.txt".format(data_dir, lang1, lang2)
        DIR_TEST_DICT = "{}/{}-{}.5000-6500.txt".format(data_dir, lang1, lang2)

    os.makedirs(DIR_NEW, exist_ok=True)
    train_dir = os.path.join(DIR_NEW, "{}2{}_train.txt".format(lang1,lang2))

    # output_dir = output_root + "mbert_{}2{}_{}".format(lang1,lang2,train_size)
    output_dir = output_root + "labse_{}2{}_{}{}".format(lang1,lang2,train_size, output_dir_postfix)
    override_output = False

    #GENERATE_NEGATIVE_SAMPLES = True
    GENERATE_NEGATIVE_SAMPLES = False
    #TRAIN_LM = True
    TRAIN_LM = False
    EVALUATE = True

    if GENERATE_NEGATIVE_SAMPLES:
      print("GEN NEG SAMPLES")
      sys.stdout.flush()
      os.system('CUDA_VISIBLE_DEVICES={} python gen_neg_samples.py --l1 {} --l2 {} --train_size {} --root {} --num_neg {} --neg_max {} --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --train_dict_dir {} --test_dict_dir {}'.format(gpuid,lang1, lang2, train_size, DIR_NEW, num_neg, neg_max, l1_voc, l1_emb, l2_voc, l2_emb, DIR_TRAIN_DICT, DIR_TEST_DICT))

    if TRAIN_LM:
      print("C2 Contrastive TRAINING")
      if os.path.isdir(output_dir):
        print(f"[WARNING] {output_dir} already exists!")
        if override_output:
          print(f"[WARNING] Removing {output_dir}!")
          os.system("rm -f {}/*".format(output_dir))
        else:
          output_dir = f"{output_dir}_{time.time()}"
          print(f"[WARNING] New output directory is: {output_dir}")

      # else:
      print(f"[INFO] Creating {output_dir}")
      os.system("mkdir -p {}".format(output_dir))

      sys.stdout.flush() 
      os.system("CUDA_VISIBLE_DEVICES={} python3 train.py --model_dir {} --train_dir {} --output_dir {} --l1 {} --l2 {} --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --use_cuda --epoch {} --train_batch_size {} --learning_rate {} --max_length {} --checkpoint_step {} --parallel --amp --random_seed {} --infoNCE_tau {} --random_erase 0 --dropout_rate 0.1 --agg_mode {} --num_neg {} --template {} --random_seed {}".format(gpuid, model_dir, train_dir, output_dir, lang1, lang2, l1_voc, l1_emb, l2_voc, l2_emb, epoch, train_batch_size, learning_rate, max_length, checkpoint_step, random_seed, infoNCE_tau, agg_mode, num_neg, template, random_seed))

    if EVALUATE:
      print("EVALUATION")
      sys.stdout.flush() 
      os.system("CUDA_VISIBLE_DEVICES={} python evaluate_bli_procrustes.py --l1 {} --l2 {} --train_size {} --root {} --model_name {} --agg_mode {} --template {} --max_length {} --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --train_dict_dir {} --test_dict_dir {} --lambda_ {} --origin_model_name {} --top_k 1,5,10".format(gpuid, lang1, lang2, train_size, DIR_NEW, output_dir, agg_mode, template, max_length,l1_voc, l1_emb, l2_voc, l2_emb, DIR_TRAIN_DICT, DIR_TEST_DICT, lambda_, model_dir))

