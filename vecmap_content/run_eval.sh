#binn/bash

s=${1:-en}
#s=${1:-fr}
#s=${1:-si}
#s=${1:-ta}
#s=${1:-es}
#s=${1:-tr}

#t=${2:-en}
#t=${2:-es}
t=${2:-tr}
#t=${2:-fr}
#t=${2:-si}
#t=${2:-ta}

base="wiki"
#base="cc"

#retrieval="nn"
#retrieval="invnn"
#retrieval="invsoftmax"
retrieval="csls"

# aligned embeddings in src --> tgt is used to evaluate tgt --> src as well. 
# set swap_embed_src_tgt_names=True if aligned embeddings are created using src --> tgt but evaluating in tgt --> src direction
if [[ ${s} == "en" ]]; then
  swap_embed_src_tgt_names="False"
  #swap_embed_src_tgt_names="True"
else
  swap_embed_src_tgt_names="True"
  #swap_embed_src_tgt_names="False"
fi

#prune_ascii="True"
prune_ascii="False"

alignment="sup"  # supervised
#alignment="unsup"  # unsupervised

if [[ ${s} == "si" || ${t} == "si" ]]; then
  #dico_train=data/wiki_maximum_trainset_5000_200k.txt
  #dico_test=data/wiki_maximum_testset_1500_200k.txt

  dico_train="data/${base}_${s}-${t}_trainset_1500_5000_200k.txt"
  dico_test="data/${base}_${s}-${t}_testset_1500_5000_200k.txt"
else

  dico_train=data/${s}-${t}.0-5000.txt
  dico_test=data/${s}-${t}.5000-6500.txt
fi

if [ ! -f "${dico_train}" ]; then
  DICO=$(basename -- "${dico_train}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

if [ ! -f "${dico_test}" ]; then
  DICO=$(basename -- "${dico_test}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

echo "[INFO] Testset: ${dico_test}"

if [ $base == "cc" ]; then
  if [ $swap_embed_src_tgt_names == "True" ]; then
    src_emb_mapped=data/${base}.${s}.mapped_${t}_${s}_${alignment}.300.vec
    tgt_emb_mapped=data/${base}.${t}.mapped_${t}_${s}_${alignment}.300.vec
  else
    src_emb_mapped=data/${base}.${s}.mapped_${s}_${t}_${alignment}.300.vec
    tgt_emb_mapped=data/${base}.${t}.mapped_${s}_${t}_${alignment}.300.vec
  fi

else
  if [ $swap_embed_src_tgt_names == "True" ]; then
    src_emb_mapped=data/${base}.${s}.mapped_${t}_${s}_${alignment}.vec
    tgt_emb_mapped=data/${base}.${t}.mapped_${t}_${s}_${alignment}.vec
  else
    src_emb_mapped=data/${base}.${s}.mapped_${s}_${t}_${alignment}.vec
    tgt_emb_mapped=data/${base}.${t}.mapped_${s}_${t}_${alignment}.vec
  fi

fi

echo "[INFO] src embeddings: ${src_emb_mapped}"
echo "[INFO] tgt embeddings: ${src_emb_mapped}"

echo "[INFO] Evaluating the aligned embeddings using BLI ${s} --> ${t} ${retrieval} retrieval"

if [ $prune_ascii == "True" ]; then
  python eval_translation.py ${src_emb_mapped} ${tgt_emb_mapped} -d ${dico_test} --retrieval ${retrieval} --prune_ascii
else
  python eval_translation.py ${src_emb_mapped} ${tgt_emb_mapped} -d ${dico_test} --retrieval ${retrieval}
fi
