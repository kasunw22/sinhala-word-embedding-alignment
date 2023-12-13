#bin/bash
s="en"
t="tr"

#base="cc"
base="wiki"

exp_name="$train_{s}_${t}"

n_refinement=0

#export="txt"
export="pth"

dico_max_rank=0  # 0 to disable (default 10000)

### alignment datasets
if [[ ${s} == "si" || ${t} == "si" ]]; then
        dico_train=../all_data/${base}_${s}-${t}_trainset_1500_5000_200k.txt
        dico_test=../all_data/${base}_${s}-${t}_testset_1500_5000_200k.txt
else
        dico_train=../all_data/${s}-${t}.0-5000.txt
        dico_test=../all_data/${s}-${t}.5000-6500.txt

        if [ ! -f "${dico_train}" ]; then
                DICO=$(basename -- "${dico_train}")
                wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P ../all_data/
        fi
        if [ ! -f "${dico_test}" ]; then
                DICO=$(basename -- "${dico_test}")
                wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P ../all_data/
        fi
fi

### embedding model
if [ $base == "cc" ]; then

  src_emb_gz=../all_data/${base}.${s}.300.vec.gz
  src_emb=../all_data/${base}.${s}.300.vec
  if [ ! -f "${src_emb}" ]; then
    EMB=$(basename -- "${src_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${src_emb_gz}
  fi

  tgt_emb_gz=../all_data/${base}.${t}.300.vec.gz
  tgt_emb=../all_data/${base}.${t}.300.vec
  if [ ! -f "${tgt_emb}" ]; then
    EMB=$(basename -- "${tgt_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${tgt_emb_gz}
  fi

else

  src_emb=../all_data/${base}.${s}.vec
  if [ ! -f "${src_emb}" ]; then
    curl -Lo ${src_emb} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${s}.vec
  fi

  tgt_emb=../all_data/${base}.${t}.vec
  if [ ! -f "${tgt_emb}" ]; then
    curl -Lo ${tgt_emb} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${t}.vec
  fi

fi

### alignment
python supervised.py --src_lang ${s} --tgt_lang ${t} --src_emb ${src_emb} --tgt_emb ${tgt_emb} --n_refinement ${n_refinement} --cuda False --dico_train ${dico_train} --dico_eval ${dico_test} --normalize_embeddings renorm,center,renorm --exp_name ${exp_name} --export ${export} --dico_max_rank ${dico_max_rank}

# supervised en-si ===================================================================

# cc-5k
#python supervised.py --src_lang en --tgt_lang si --src_emb ../all_data/cc.en.300.vec --tgt_emb ../all_data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../all_data/cc_en-si_trainset_1500_5000_200k.txt --dico_eval ../all_data/cc_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_sicc_5k --export ${export} --dico_max_rank ${dico_max_rank}

# wiki-5k
#python supervised.py --src_lang en --tgt_lang si --src_emb ../all_data/wiki.en.vec --tgt_emb ../all_data/wiki.si.vec --n_refinement ${n_refinement} --cuda False --dico_train ../all_data/wiki_en-si_trainset_1500_5000_200k.txt --dico_eval ../all_data/wiki_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_wiki_5k --export ${export} --dico_max_rank ${dico_max_rank}

# cc-full
#python supervised.py --src_lang en --tgt_lang si --src_emb ../all_data/cc.en.300.vec --tgt_emb ../all_data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../all_data/cc_en-si-full_trainset_wo_test_1500.txt --dico_eval ../all_data/cc_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_cc_full --export ${export} --dico_max_rank ${dico_max_rank}

# wiki-full
#python supervised.py --src_lang en --tgt_lang si --src_emb ../all_data/cc.en.300.vec --tgt_emb ../all_data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../all_data/wiki_en-si-full_trainset_wo_test_1500.txt --dico_eval ../all_data/wiki_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_wiki_full --export ${export} --dico_max_rank ${dico_max_rank}

# prob-based-dict
#python supervised.py --src_lang en --tgt_lang si --src_emb ../all_data/cc.en.300.vec --tgt_emb ../all_data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../all_data/full-prob-en-si-cc-osub-count_once-False.txt --dico_eval ../all_data/cc_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_prob_dict --export ${export} --dico_max_rank ${dico_max_rank}

#supervised si-en ======================================================================

# cc-5k
#python supervised.py --src_lang si --tgt_lang en --src_emb ../all_data/cc.si.300.vec --tgt_emb ../all_data/cc.en.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../all_data/cc_si-en_trainset_1500_5000_200k.txt --dico_eval ../all_data/cc_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name si_en-new --export ${export} --dico_max_rank ${dico_max_rank}

# wiki-5k
#python supervised.py --src_lang si --tgt_lang en --src_emb ../all_data/wiki.si.vec --tgt_emb ../all_data/wiki.en.vec --n_refinement ${n_refinement} --cuda False --dico_train ../all_data/wiki_si-en_trainset_1500_5000_200k.txt --dico_eval ../all_data/wiki_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name si_en-new --export ${export} --dico_max_rank ${dico_max_rank}

#python supervised.py --src_lang en --tgt_lang hi --src_emb ../all_data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement ${n_refinement} --dico_train default --cuda False --dico_eval default --export ${export} --dico_max_rank ${dico_max_rank}
