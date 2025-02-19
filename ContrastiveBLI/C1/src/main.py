import sys
import io
import subprocess as commands
import codecs
import copy
import argparse
import math
import pickle as pkl
import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from util import *
from models import *
from dataloader import *
from forward import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tqdm

random = np.random
random.seed(1234)

def normed_input(x):
    y = x/(np.linalg.norm(x,axis=1,keepdims=True) + 1e-9)
    return y


def getknn(src_, tgt_, tgt_ids, k=10, bsz=1024):
    src_ = src_.cuda()
    tgt_ = tgt_.cuda()
    src = src_ / (torch.norm(src_, dim=1, keepdim=True) + 1e-9)
    tgt = tgt_ / (torch.norm(tgt_, dim=1, keepdim=True) + 1e-9)
    num_imgs = len(src)
    confuse_output_indices = []
    confuse_output_indices_long = []
    for batch_idx in range( int( math.ceil( float(num_imgs) / bsz ) ) ):
        start_idx = batch_idx * bsz
        end_idx = min( num_imgs, (batch_idx + 1) * bsz )
        length = end_idx - start_idx
        prod_batch = torch.matmul(src[start_idx:end_idx, :], tgt.T)
        dotprod = torch.topk(prod_batch,k=k+1,dim=1,sorted=True,largest=True).indices
        confuse_output_indices_long += dotprod.cpu().tolist()


    for i in range(len(confuse_output_indices_long)):
        confuse_output_i = confuse_output_indices_long[i]
        if tgt_ids[i] in confuse_output_i:
            confuse_output_i_new = confuse_output_i.copy()
            confuse_output_i_new.remove(tgt_ids[i])
            confuse_output_indices.append(confuse_output_i_new)
        else:
            confuse_output_indices.append(confuse_output_i[:-1])
    return confuse_output_indices

  
def remove_dup(con, con_s):
    new_con = []
    batch = len(con)
    assert len(con_s) == batch
    for i in range(batch):
        set_s = set(con_s[i])
        new_b = []
        for item in con[i]:
            if item not in set_s:
                new_b.append(item)
       
        new_con.append(new_b[:20])
        assert len(new_con[-1])==20 
    return new_con

def neg_sample(model,args,train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup):
    batch_size = args.valid_batch_size
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)
    neg_sample = args.num_sample
    neg_max = 60000
    for batch_idx in range( int( math.ceil( float(num_imgs_l1) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l1, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l1[start_idx:end_idx].cuda()
        src_mid = model.eval_src2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l1_translation = src_mid
        else:
            train_data_l1_translation = torch.cat([train_data_l1_translation,src_mid],dim = 0)
    train_data_l1_translation = train_data_l1_translation.cpu()
    sup_data_l1_translation = torch.index_select(train_data_l1_translation,0,torch.tensor(l1_idx_sup))

    for batch_idx in range( int( math.ceil( float(num_imgs_l2) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l2, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l2[start_idx:end_idx].cuda()
        tgt_mid = model.eval_tgt2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l2_translation = tgt_mid
        else:
            train_data_l2_translation = torch.cat([train_data_l2_translation,tgt_mid],dim = 0)
    train_data_l2_translation = train_data_l2_translation.cpu()
    sup_data_l2_translation = torch.index_select(train_data_l2_translation,0,torch.tensor(l2_idx_sup))

    confuse_tgt = getknn(sup_data_l1_translation, train_data_l2_translation[:neg_max], l2_idx_sup, k = neg_sample, bsz=1024) 
    confuse_src = getknn(sup_data_l2_translation, train_data_l1_translation[:neg_max], l1_idx_sup, k = neg_sample, bsz=1024)  


    confuse_tgt_tensor = torch.tensor(confuse_tgt)
    confuse_src_tensor = torch.tensor(confuse_src)

    confuse_tgt = torch.cat([torch.tensor(l2_idx_sup).unsqueeze(1) , confuse_tgt_tensor],dim=1) 
    confuse_src = torch.cat([torch.tensor(l1_idx_sup).unsqueeze(1) , confuse_src_tensor],dim=1) 
    return confuse_src, confuse_tgt


def valid_BLI(model, args, train_data_l1, train_data_l2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s):
    batch_size = args.valid_batch_size
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)

    for batch_idx in range( int( math.ceil( float(num_imgs_l1) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l1, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l1[start_idx:end_idx].cuda()
        src_mid = model.eval_src2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l1_translation = src_mid
        else:
            train_data_l1_translation = torch.cat([train_data_l1_translation,src_mid],dim = 0)

    for batch_idx in range( int( math.ceil( float(num_imgs_l2) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l2, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l2[start_idx:end_idx].cuda()
        tgt_mid = model.eval_tgt2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l2_translation = tgt_mid
        else:
            train_data_l2_translation = torch.cat([train_data_l2_translation,tgt_mid],dim = 0)

    acc_s2t = compute_nn_accuracy_torch(train_data_l1_translation, train_data_l2_translation, src2tgt, args, model, lexicon_size=-1) 
    cslsacc_s2t = compute_csls_accuracy(train_data_l1_translation, train_data_l2_translation, src2tgt, args, model, lexicon_size=-1)

    acc_t2s = compute_nn_accuracy_torch(train_data_l2_translation, train_data_l1_translation, tgt2src, args, model, lexicon_size=-1) 
    cslsacc_t2s = compute_csls_accuracy(train_data_l2_translation, train_data_l1_translation, tgt2src, args, model, lexicon_size=-1)

    BLI_accuracy_l12l2 = (acc_s2t, cslsacc_s2t)
    BLI_accuracy_l22l1 = (acc_t2s, cslsacc_t2s)
    return (BLI_accuracy_l12l2, BLI_accuracy_l22l1) 

 
def high_conf_pairs(model, args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup):
    batch_size = args.valid_batch_size
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)

    for batch_idx in range( int( math.ceil( float(num_imgs_l1) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l1, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l1[start_idx:end_idx].cuda()
        src_mid = model.eval_src2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l1_translation = src_mid
        else:
            train_data_l1_translation = torch.cat([train_data_l1_translation,src_mid],dim = 0)

    for batch_idx in range( int( math.ceil( float(num_imgs_l2) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l2, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l2[start_idx:end_idx].cuda()
        tgt_mid = model.eval_tgt2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l2_translation = tgt_mid
        else:
            train_data_l2_translation = torch.cat([train_data_l2_translation,tgt_mid],dim = 0)
 
    l1_idx_aug, l2_idx_aug = generate_new_dictionary_bidirectional(args, train_data_l1_translation, train_data_l2_translation, l1_idx_sup, l2_idx_sup)



    return l1_idx_aug, l2_idx_aug


def SAVE_DATA(model, args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup, voc_l1, voc_l2):
    batch_size = args.valid_batch_size
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)

    for batch_idx in range( int( math.ceil( float(num_imgs_l1) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l1, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l1[start_idx:end_idx].cuda()
        src_mid = model.eval_src2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l1_translation = src_mid
        else:
            train_data_l1_translation = torch.cat([train_data_l1_translation,src_mid],dim = 0)

    for batch_idx in range( int( math.ceil( float(num_imgs_l2) / batch_size ) ) ):
        start_idx = batch_idx * batch_size
        end_idx = min( num_imgs_l2, (batch_idx + 1) * batch_size )
        length = end_idx - start_idx
        input_ = train_data_l2[start_idx:end_idx].cuda()
        tgt_mid = model.eval_tgt2mid(input_, mode='valid')
        if batch_idx == 0:
            train_data_l2_translation = tgt_mid
        else:
            train_data_l2_translation = torch.cat([train_data_l2_translation,tgt_mid],dim = 0)

    train_data_l1_translation = train_data_l1_translation.cpu()
    train_data_l2_translation = train_data_l2_translation.cpu()

    sup_data_l1_translation = torch.index_select(train_data_l1_translation,0,torch.tensor(l1_idx_sup))
    sup_data_l2_translation = torch.index_select(train_data_l2_translation,0,torch.tensor(l2_idx_sup))

    s_l = "1k" if args.train_size == "1k" else "5k"

    np.save(args.save_dir + "{}/{}2{}_{}_voc.npy".format(s_l, args.l1, args.l2, args.l1), voc_l1)
    np.save(args.save_dir + "{}/{}2{}_{}_voc.npy".format(s_l, args.l1, args.l2, args.l2), voc_l2)
    torch.save(train_data_l1_translation, args.save_dir + "{}/{}2{}_{}_emb.pt".format(s_l, args.l1, args.l2, args.l1)) #aligned l1 WEs
    torch.save(train_data_l2_translation, args.save_dir + "{}/{}2{}_{}_emb.pt".format(s_l, args.l1, args.l2, args.l2)) #aligned l2 WEs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C1: Contrastive Linear Mapping')

    parser.add_argument("--l1", type=str, default=" ",
                    help="Language l1 (string)")
    parser.add_argument("--l2", type=str, default=" ",
                    help="Language l2 (string)")
    parser.add_argument("--num_games", type=int, default=200,
                    help="Total number of training steps")
    parser.add_argument("--num_sl", type=int, default=20,
                    help="Total number of self-learninig loops")
    parser.add_argument("--sup_batch_size", type=int, default=5000,
                    help="Batch size For Training")
    parser.add_argument("--mini_batch_size", type=int, default=5000,
                    help="Batch size For Training")
    parser.add_argument("--valid_batch_size", type=int, default=5000,
                    help="Batch size For Validation")
    parser.add_argument("--D_emb", type=int, default=300,
                    help="Pretrained static word embedding dimensionality")
    parser.add_argument("--lr", type=float, default=1.5,
                    help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                    help="gamma")
    parser.add_argument("--dropout", type=float, default=0,
                    help="Dropout keep probability")
    parser.add_argument("--print_every", type=int, default=25,
                    help="Print every k training steps")
    parser.add_argument("--valid_every", type=int, default=50,
                    help="Validate model every k training steps")
    parser.add_argument("--grad_clip", type=float, default=0.15,
                    help="Clip gradient")
    parser.add_argument("--norm_input", action="store_true", default=True,
                    help="True if unit-norm word embeddings")
    parser.add_argument("--num_sample", type=int, default=150,
                    help="Number of hard negative samples in contrastive training objective")
    parser.add_argument("--neg_max", type=int, default=60000,
                    help="Number of hard negative samples in contrastive training objective")
    parser.add_argument("--dico_max_rank", type=int, default=20000,
                    help="Number of hard negative samples in contrastive training objective")
    parser.add_argument("--resnet", action="store_true", default=False,
                    help="Add residual connection to linear mappings")
    parser.add_argument("--cpu", action="store_true", default=False,
                    help="True if using cpu only")
    parser.add_argument("--self_learning", action="store_true", default=True,
                    help="True if using 1k translation pairs and do self-learning to augment training samples")
    parser.add_argument("--save_aligned_we", action="store_true", default=False,
                    help="True if saving aligned WEs")    
    parser.add_argument("--num_aug", type=int, default=6000,
                    help="Print every k training steps")
    parser.add_argument("--train_size", type=str, default="1k",
                    help="train dict size")
    parser.add_argument("--emb_src_dir", type=str, default="./",
                    help="emb_src_dir")
    parser.add_argument("--tgt_src_dir", type=str, default="./",
                    help="tgt_src_dir")
    parser.add_argument("--train_dict_dir", type=str, default="./",
                    help="train_dict_dir")
    parser.add_argument("--test_dict_dir", type=str, default="./",
                    help="test_dict_dir")    
    parser.add_argument("--save_dir", type=str, default="./",
                    help="save_dir")


    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")

    args.str2lang = {"hr":"croatian", "en":"english","fi":"finnish","fr":"french","de":"german","it":"italian","ru":"russian","tr":"turkish", "es": "spanish", "si": "sinhala"}
    print("C1 Model: {}({}) <---> {}({})".format( args.str2lang[args.l1],args.l1,args.str2lang[args.l2],args.l2))


    # The Two Settings for Reproducing Our Reported Results:
    if args.train_size == "1k":
        args.sup_batch_size = 1000
        args.mini_batch_size = 1000
        args.num_sample = 60
        args.lr = 2.0
        args.num_games = 50
        args.print_every = 1000000
        args.num_sl = 3
        args.num_aug = 6000
        args.dico_max_rank = 20000
        args.gamma = 1.0
        args.valid_every = 50
    elif args.train_size == "5k":
        args.sup_batch_size = 5000
        args.mini_batch_size = 5000
        args.num_sample = 150
        args.lr = 1.5
        args.num_games = 200
        args.print_every = 1000000
        args.num_sl = 2
        args.num_aug = 10000
        args.dico_max_rank = 60000
        args.gamma = 0.99
        args.valid_every = 200
    else:
        print("Unknown Setting, Please Conduct Hyperparameter Search on Your Dataset.")


####Defining Directories
    DIR_EMB_SRC = args.emb_src_dir
    DIR_EMB_TRG = args.tgt_src_dir
    DIR_TEST_DICT = args.test_dict_dir
    DIR_TRAIN_DICT = args.train_dict_dir



####LOAD WORD EMBS
    voc_l1, embs_l1 = load_embs(DIR_EMB_SRC)#,topk=10000)
    print("L1 INPUT WORD VECTOR SPACE OF SIZE:", embs_l1.shape)

    args.class_num_l1 = len(voc_l1)
    print("L1 Contain", args.class_num_l1, " Words")

    voc_l2, embs_l2 = load_embs(DIR_EMB_TRG)#,topk=10000)
    print("L2 INPUT WORD VECTOR SPACE OF SIZE:", embs_l2.shape)
    args.class_num_l2 = len(voc_l2)
    print("L2 Contain", args.class_num_l2, " Words")

    if args.norm_input:
        embs_l1 = normed_input(embs_l1)
        embs_l2 = normed_input(embs_l2)

    train_data_l1 = torch.from_numpy(embs_l1.copy())
    train_data_l2 = torch.from_numpy(embs_l2.copy())
    print("Static WEs Loaded")

#### LOAD TRAIN PARALLEL DATA
    file = open(DIR_TRAIN_DICT,'r')
    l1_dic = []
    l2_dic = []
    for line in file.readlines():
        # pair = line[:-1].split('\t')
        pair = line[:-1].split()
        l1_dic.append(pair[0].lower())
        l2_dic.append(pair[1].lower())
    file.close()
    l1_idx_sup = []
    l2_idx_sup = []

    train_pairs = set()

    for i in range(len(l1_dic)):
        l1_tok = voc_l1.get(l1_dic[i])
        l2_tok = voc_l2.get(l2_dic[i])
        if (l1_tok is not None) and (l2_tok is not None):
            l1_idx_sup.append(l1_tok)
            l2_idx_sup.append(l2_tok)
            train_pairs.add((l1_tok,l2_tok))

    print("Sup Set Size: ", len(l1_idx_sup), len(l2_idx_sup))
    print("Sup L1 Word Frequency Ranking: ", 'min ',min(l1_idx_sup), ' max ', max(l1_idx_sup), ' average ', float(sum(l1_idx_sup))/len(l1_idx_sup))
    print("Sup L2 Word Frequency Ranking: ", 'min ',min(l2_idx_sup), ' max ', max(l2_idx_sup), ' average ', float(sum(l2_idx_sup))/len(l2_idx_sup))
    sys.stdout.flush() 

    

#### LOAD TEST DATA
    words_src = list(voc_l1.keys())
    words_tgt = list(voc_l2.keys())
    print(idx(words_src)==voc_l1,idx(words_tgt)==voc_l2)
    src2tgt, lexicon_size_s2t = load_lexicon_s2t(DIR_TEST_DICT, words_src, words_tgt)
    tgt2src, lexicon_size_t2s = load_lexicon_t2s(DIR_TEST_DICT, words_tgt, words_src)
    print("lexicon_size_s2t, lexicon_size_t2s", lexicon_size_s2t, lexicon_size_t2s)
    del words_src, words_tgt
    #####

    print(args)


    model = C1_Model(args)
    print(model)
    if not args.cpu:
        model = model.cuda()

    in_params = []
    in_names = []
    for name, param in model.named_parameters():
        in_params.append(param)
        in_names.append(name)


    print("in_params: ", in_names)
    sys.stdout.flush()
    in_size = [x.size() for x in in_params]
    in_sum = sum([np.prod(x) for x in in_size])

    optimizer = torch.optim.SGD(in_params, lr=args.lr)
    train_loss_dict_ = get_log_loss_dict_()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    
    print("BLI Prediction Accuracy in (NN Retrieval, CSLS Retrieval) format:")
    if args.self_learning:
        print("Self-Learning Mode!")

        for iter_sl in range(args.num_sl):
            l1_idx_aug = []
            l2_idx_aug = []
            if iter_sl > 0:
                with torch.no_grad():
                    model.eval()
                    l1_idx_aug, l2_idx_aug = high_conf_pairs(model, args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup)
                    print("Iteration: ", iter_sl, "augment ", len(l1_idx_aug), " training pairs")
                    sys.stdout.flush()
                model.train()


            args.sup_batch_size = len(l1_idx_sup) + len(l1_idx_aug)
            args.mini_batch_size = len(l2_idx_sup) + len(l2_idx_aug)

            l1_idx_current = l1_idx_sup+l1_idx_aug
            l2_idx_current = l2_idx_sup+l2_idx_aug

            src_mid_transform, trg_mid_transform = AdvancedMapping(embs_l1.copy()[l1_idx_current,:],embs_l2.copy()[l2_idx_current,:])

        
            s2m_mapping = torch.from_numpy(src_mid_transform)
            t2m_mapping = torch.from_numpy(trg_mid_transform)
        
            if args.resnet:
                s2m_mapping = s2m_mapping - torch.eye(args.D_emb)
                t2m_mapping = t2m_mapping - torch.eye(args.D_emb)

            pdict = {'s2m_mapping.weight':s2m_mapping.cuda(), 't2m_mapping.weight':t2m_mapping.cuda()}
            model_dict=model.state_dict()
            pretrained_dict = {}
            for k,v in model_dict.items():
                if k in pdict:
                    pretrained_dict[k] = pdict[k]
                else:
                    pretrained_dict[k] = model_dict[k]
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
         
            if args.train_size == "5k":
                l1_idx_current = l1_idx_sup
                l2_idx_current = l2_idx_sup
                args.sup_batch_size = len(l1_idx_sup)
                args.mini_batch_size = len(l2_idx_sup)
            optimizer.param_groups[0]['lr'] = args.lr    
            for epoch in range(args.num_games+1):
                if epoch ==0: 
                    with torch.no_grad():
                        model.eval()
                        accuracy_BLI = valid_BLI(model, args, train_data_l1, train_data_l2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s)
                        print("(BEFORE TRAINING)", " Iter: ", iter_sl, " Epoch: ", epoch, "BLI Accuracy L1 to L2: ", accuracy_BLI[0], "BLI Accuracy L2 to L1: ", accuracy_BLI[1])
                        sys.stdout.flush()
                    model.train()
    
                if epoch % 1 == 0:
                    with torch.no_grad():
                        model.eval()
                        confuse_src, confuse_tgt = neg_sample(model,args,train_data_l1, train_data_l2, l1_idx_current, l2_idx_current)
                    model.train()
    
                src_input, tgt_input = next_batch_joint_sup_confuse(confuse_src, confuse_tgt, train_data_l1, train_data_l2, args.sup_batch_size, args)
    
    
    
                optimizer.zero_grad()
    
                for i in range(args.sup_batch_size//args.mini_batch_size):
    
                    src_input_mini = src_input[i*args.mini_batch_size:(i+1)*args.mini_batch_size,:,:].cuda()
                    tgt_input_mini = tgt_input[i*args.mini_batch_size:(i+1)*args.mini_batch_size,:,:].cuda()
                    loss = forward_joint((src_input_mini,tgt_input_mini), model, train_loss_dict_, args, mode="train")
                    loss.backward()
    
    
                total_norm = nn.utils.clip_grad_norm(in_params, args.grad_clip)
                optimizer.step()
    
    
                if epoch % args.print_every == 0:
                    avg_loss_dict_ = get_avg_from_loss_dict_(train_loss_dict_)
                    print(print_loss_(epoch, avg_loss_dict_, "train"))
                    sys.stdout.flush()
                    train_loss_dict_ = get_log_loss_dict_()
    
                valid_or_not = True
                if (epoch % args.valid_every == 0) and valid_or_not and (epoch > 0):
    
                    with torch.no_grad():
                        model.eval()
                        accuracy_BLI = valid_BLI(model, args, train_data_l1, train_data_l2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s)
                        print("C1", " Iter: ", iter_sl,  " Epoch: ", epoch, "BLI Accuracy L1 to L2: ", accuracy_BLI[0], "BLI Accuracy L2 to L1: ", accuracy_BLI[1], " {}({}) <---> {}({})".format( args.str2lang[args.l1],args.l1,args.str2lang[args.l2],args.l2))
                        sys.stdout.flush()
                    model.train()
                scheduler.step()

    
    #Finally Save Aligned WEs
    if args.save_aligned_we:
        with torch.no_grad():
            model.eval() 
            SAVE_DATA(model, args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup, voc_l1, voc_l2)
        print("Data Saved")


    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
