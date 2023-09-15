#bin/bash

n_refinement=0

#export="txt"
export="pth"

dico_max_rank=0  # 0 to disable (default 10000)

# supervised en-si ===================================================================

# cc-5k
#python supervised.py --src_lang en --tgt_lang si --src_emb ../msc/data/cc.en.300.vec --tgt_emb ../msc/data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../msc/data/cc_en-si_trainset_1500_5000_200k.txt --dico_eval ../msc/data/cc_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_sicc_5k --export ${export} --dico_max_rank ${dico_max_rank}

# wiki-5k
#python supervised.py --src_lang en --tgt_lang si --src_emb ../msc/data/wiki.en.vec --tgt_emb ../msc/data/wiki.si.vec --n_refinement ${n_refinement} --cuda False --dico_train ../msc/data/wiki_en-si_trainset_1500_5000_200k.txt --dico_eval ../msc/data/wiki_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_wiki_5k --export ${export} --dico_max_rank ${dico_max_rank}

# cc-full
python supervised.py --src_lang en --tgt_lang si --src_emb ../msc/data/cc.en.300.vec --tgt_emb ../msc/data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../msc/data/cc_en-si-full_trainset_wo_test_1500.txt --dico_eval ../msc/data/cc_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_cc_full --export ${export} --dico_max_rank ${dico_max_rank}

# wiki-full
#python supervised.py --src_lang en --tgt_lang si --src_emb ../msc/data/cc.en.300.vec --tgt_emb ../msc/data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../msc/data/wiki_en-si-full_trainset_wo_test_1500.txt --dico_eval ../msc/data/wiki_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_wiki_full --export ${export} --dico_max_rank ${dico_max_rank}

# prob-based-dict
#python supervised.py --src_lang en --tgt_lang si --src_emb ../msc/data/cc.en.300.vec --tgt_emb ../msc/data/cc.si.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../msc/data/full-prob-en-si-cc-osub-count_once-False.txt --dico_eval ../msc/data/cc_en-si_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name en_si_prob_dict --export ${export} --dico_max_rank ${dico_max_rank}

#supervised si-en ======================================================================

# cc-5k
#python supervised.py --src_lang si --tgt_lang en --src_emb ../msc/data/cc.si.300.vec --tgt_emb ../msc/data/cc.en.300.vec --n_refinement ${n_refinement} --cuda False --dico_train ../msc/data/cc_si-en_trainset_1500_5000_200k.txt --dico_eval ../msc/data/cc_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name si_en-new --export ${export} --dico_max_rank ${dico_max_rank}

# wiki-5k
#python supervised.py --src_lang si --tgt_lang en --src_emb ../msc/data/wiki.si.vec --tgt_emb ../msc/data/wiki.en.vec --n_refinement ${n_refinement} --cuda False --dico_train ../msc/data/wiki_si-en_trainset_1500_5000_200k.txt --dico_eval ../msc/data/wiki_si-en_testset_1500_5000_200k.txt --normalize_embeddings renorm,center,renorm --exp_name si_en-new --export ${export} --dico_max_rank ${dico_max_rank}

#python supervised.py --src_lang en --tgt_lang hi --src_emb ../msc/data/wiki.en.vec --tgt_emb data/wiki.hi.vec --n_refinement ${n_refinement} --dico_train default --cuda False --dico_eval default --export ${export} --dico_max_rank ${dico_max_rank}
