#bin/bash

#python evaluate.py --src_lang si --tgt_lang en --src_emb dumped/si_en-new/aqnm52csk7/vectors-si.txt --tgt_emb dumped/si_en-new/aqnm52csk7/vectors-en.txt --max_vocab 200000 --exp_path eval_si_en --cuda False --dico_eval ../msc/data/cc_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm

python evaluate.py --src_lang si --tgt_lang en --src_emb dumped/si_en-new/aqnm52csk7/vectors-si.txt --tgt_emb dumped/si_en-new/aqnm52csk7/vectors-en.txt --max_vocab 200000 --exp_path eval_si_en --cuda False --dico_eval ../msc/data/wiki_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm
