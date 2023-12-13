#bin/bash

s="en"
t="ta"

src_emb=x
tgt_emb=y

#base="cc"
base="wiki"

exp_name="eval_${s}_${t}"

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

python evaluate.py --src_lang ${src} --tgt_lang ${t} --src_emb ${src_emb} --tgt_emb ${tgt_emb} --max_vocab 200000 --exp_path ${exp_name} --cuda False --dico_eval ${dico_test} --normalize_embeddings renorm,center,renorm

#python evaluate.py --src_lang si --tgt_lang en --src_emb dumped/si_en-new/aqnm52csk7/vectors-si.txt --tgt_emb dumped/si_en-new/aqnm52csk7/vectors-en.txt --max_vocab 200000 --exp_path eval_si_en --cuda False --dico_eval ../all_data/cc_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm

#python evaluate.py --src_lang si --tgt_lang en --src_emb dumped/si_en-new/aqnm52csk7/vectors-si.txt --tgt_emb dumped/si_en-new/aqnm52csk7/vectors-en.txt --max_vocab 200000 --exp_path eval_si_en --cuda False --dico_eval ../all_data/wiki_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm
