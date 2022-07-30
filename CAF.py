import argparse
import copy
import math
import os
import os.path as osp
import pdb
import random
import sys
from collections import Counter
from itertools import *
from operator import le
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torch._C import device
from torch.nn.modules import padding
from torch.nn.modules.activation import PReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import loss
import network
from data_list import ImageList, ImageList_idx, Listset
from infer_semantics_and_obtain_UTR import infer_semantics_and_obtain_UTR
from loss import CrossEntropy



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer





def image_train2(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        #transforms.RandomRotation(10),
        transforms.RandomCrop(crop_size),
        #transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(0.3,0.3,0.3,0.01),
        #RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize
    ])

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(0.7,0.7,0.7,0.01),
        transforms.ToTensor(),
        normalize
    ])



def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(),transform1=image_train2())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders,dsets

def cal_acc(loader, netF_T, netB_T, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB_T(netF_T(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def load_network1(pre,network1):
    #pre=network.state_dict().copy()
    network1.load_state_dict(pre)
    pre1=network1.state_dict()
    #pre1=pre.copy()#{k:v for k,v in pre.items()}
    for k,v, in pre1.items():
        #pdb.set_trace()
            a=torch.rand_like(pre1[k],dtype=torch.float)
            a=(a-0.5)/10+1
            pre1[k]=torch.mul(pre1[k],a)
    network1.load_state_dict(pre1)
    #network.load_state_dict(pra)
    return network1





def train_target(args):
    dset_loaders,dsets = data_load(args)

    #target model
    netF_T = network.ResBase(res_name=args.net).cuda()
    netB_T = network.feat_bootleneck(type=args.classifier, feature_dim=netF_T.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    #source model
    netF_S = network.ResBase(res_name=args.net).cuda()
    netB_S = network.feat_bootleneck(type=args.classifier, feature_dim=netF_T.in_features, bottleneck_dim=args.bottleneck).cuda()
    
    #load_state_dict
    modelpath = args.output_dir_src + '/source_F.pt'   
    
    netF_T.load_state_dict(torch.load(modelpath))
    netF_S.load_state_dict(torch.load(modelpath))
    
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB_T.load_state_dict(torch.load(modelpath))
    netB_S.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))



    
    

    netC.eval()
    netB_S.eval()
    netF_S.eval()
  


    param_group = []
   
    for k, v in netF_T.named_parameters():
        param_group += [{'params': v, 'lr':args.lr*0.1}]
 
    for k, v in netB_T.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
   
    alpha=0.99
  
    optimizer= optim.SGD(param_group, weight_decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    print(len(dset_loaders["target"]))
    interval_iter = max_iter // args.interval
    print('interval:',interval_iter)
    iter_num = 0
   
    #if iter_num % interval_iter == 0 or iter_num == max_iter:
    tc=0.000001
    tp=0.9
    start=True
    
    epoch=0


    while epoch <args.max_epoch:
        epoch+=1
        
        netF_T.eval()
        netB_T.eval()
        netC.eval()
        if  start==False:
            tc=0.000001
            tp=0.9
        
       
        if start==True:
            start=False

            pre=copy.deepcopy(netF_T.state_dict())
            netF1=load_network1(pre,netF1)
            
            semantics,UTR_I,UTR_D,high_risk,have_semantics,no_semantics,data_list= infer_semantics_and_obtain_UTR(dset_loaders['test'],netF_T, netB_T, netC,args)
            semantics = semantics.cuda()
        else:
            
            k_clu=max(((len(data_list)-len(high_risk))//len(have_semantics))*2,1000)
            
            netF1=load_network1(pre,netF1)
            
            semantics,UTR_I,_,high_risk,have_semantics,no_semantics,data_list= infer_semantics_and_obtain_UTR(dset_loaders['test'],netF_T, netB_T, netC,args,k=k_clu)
          
            semantics = semantics.cuda()
          
        netF_T.train()
        netB_T.train()
        


        if len(high_risk)!=0:
            dset_high_risk=Listset(dsets["target"],high_risk)
            dset_high_risk=DataLoader(dset_high_risk, batch_size=args.batch_size//4, shuffle=True, drop_last=True)
        
        iter_num = 0
        #adaptation, mixup data augmentation is used
        while   (iter_num*(args.batch_size//4))<max(len(data_list)):
            iter_num += 1
            lr_scheduler(optimizer, iter_num=epoch, max_iter=args.max_epoch)
            
            try:
                [inputs_test,inputs_testa], label, tar_idx = iter_dset.next()
            except:
                dset=Listset(dsets["target"],data_list)
                dset=DataLoader(dset, batch_size=args.batch_size//4, shuffle=True, drop_last=True)
                iter_dset = iter(dset)
                [inputs_test,inputs_testa], label, tar_idx = iter_dset.next()
            try:
                [inputs_test1,inputs_test1a], label1, tar_idx1 = iter_dset1.next()
            except:
                data_list1=copy.deepcopy(data_list)
                random.shuffle(data_list1)
                dset1=Listset(dsets["target"],data_list1)
                dset1=DataLoader(dset1, batch_size=args.batch_size//4, shuffle=True, drop_last=True)
                iter_dset1 = iter(dset1)
                [inputs_test1,inputs_test1a], label1, tar_idx1  = iter_dset1.next()
        
            if inputs_test.size(0)>1 and inputs_test1.size(0)>1:
                
                optimizer.zero_grad()
                
                
                inputs_test = inputs_test.cuda()
                inputs_test1 = inputs_test1.cuda()
                inputs_test1a = inputs_test1a.cuda()
                #pseudo_label
                with torch.no_grad():
                    pred_u=netC(netB_T(netF_T(inputs_test1)))
                    pred_u= nn.Softmax(dim=1)(pred_u)
                    pred_u = pred_u**(1/0.5)
                    targets_u = pred_u / pred_u.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()
                
                targets_x = semantics[tar_idx]
                targets_x = torch.zeros((targets_x.size(0),12)).scatter_(1, targets_x.unsqueeze(1).cpu(), 1).cuda()
                all_inputs = torch.cat([inputs_test, inputs_test1, inputs_test1a], dim=0)

                all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
                args.alpha=0.75
                l = np.random.beta(args.alpha, args.alpha)

                l = max(l, 1-l)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                mixed_input = list(torch.split(mixed_input, args.batch_size//4))
                mixed_input = interleave(mixed_input, args.batch_size//4)
                logits = [netC(netB_T(netF_T(mixed_input[0])))]
                for input in mixed_input[1:]:
                    logits.append(netC(netB_T(netF_T(input))))
                logits = interleave(logits, args.batch_size//4)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)
                #pdb.set_trace()
                
                if  logits_u.shape[0]==32 and logits_x.shape[0]==16:
                    classifier_loss =CrossEntropy()(logits_x, mixed_target[:args.batch_size//4])
                    classifier_loss+= CrossEntropy()(logits_u,  mixed_target[args.batch_size//4:]) #
                    hplcloss=classifier_loss

                    optimizer.zero_grad()
                    hplcloss.backward()

                    optimizer.step()

               
               
            
           
        iter_num=0   
        #two calibration modules
        while iter_num*(args.batch_size//4)<len(high_risk) or iter_num*args.batch_size<len(data_list):
            iter_num+=1
            
            if len(high_risk)!=0 and iter_num*(args.batch_size//4)<len(high_risk):
                try:
                    [inputs_test,inputs_testa], label, tar_idx_risk = next(iter_high_risk)
                except:
                    dset_high_risk=Listset(dsets["target"],high_risk)
                    dset_high_risk=DataLoader(dset_high_risk, batch_size=args.batch_size, shuffle=True, drop_last=False)
                    iter_high_risk = iter((dset_high_risk))
                    [inputs_test,inputs_testa], label, tar_idx_risk = next(iter_high_risk)
                iter_hphc_num1+=1
                if inputs_test.size(0)>1:
                    pred = semantics[tar_idx_risk]
                    
                    valid=torch.nonzero(pred!=-1).squeeze()

                    pred=pred[valid]
                    if valid.shape[0]<=1:
                        continue
                    optimizer.zero_grad()
                   
                    inputs_test = inputs_test[valid].cuda()
    
                    features_test = netB_T(netF_T(inputs_test))
                    outputs_test = netC(features_test)
                    
                
                
                    forget_loss = args.w_forget* nn.CrossEntropyLoss()(outputs_test, pred)
                 
                    
                    forget_loss=args.w_forget*classifier_loss
                    optimizer.zero_grad()
                    forget_loss.backward()

                    optimizer.step()
            
            if iter_num*args.batch_size<len(data_list):
                try:
                    [inputs_test,inputs_testa], label, tar_idxlp =next( iter_discover)
                except:
                    dset=Listset(dsets["target"],data_list)
                    dset=DataLoader(dset, batch_size=args.batch_size//4, shuffle=True, drop_last=True)
                    iter_discover = iter((dset))
                    [inputs_test,inputs_testa], label, tar_idxlp = next(iter_discover)
                iter_lphc_num+=1
                if inputs_test.size(0)>1:
                    
                    optimizer.zero_grad()          
                    pred = semantics[tar_idxlp]       
                    valid=torch.nonzero(pred!=-2).squeeze()

                    if valid.shape[0]<=1:
                        continue
                    inputs_test = inputs_test[valid].cuda()
                
                    inputs_testa = inputs_testa[valid].cuda()
                    pred=pred[valid].cuda()
                    features_test = netB_T(netF_T(inputs_test))
                    outputs_test = netC(features_test)
                    
              
                    with torch.no_grad():      
                        features_testS = (netB_S(netF_S(inputs_test)))             
                    kdloss =torch.nn.MSELoss(reduce=False, size_average=False)(features_test,features_testS)
                    kdloss=torch.mean(kdloss*UTR_D) 
                    
                   
                    softmax_out = nn.Softmax(dim=1)(outputs_test)
                    discover_loss = torch.mean(loss.Entropy(softmax_out))
                    
                    msoftmax = softmax_out.mean(dim=0)
                    discover_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                    discover_loss = entropy_loss 
                 
                    if epoch>=10:
                        
                        args.w_kd=0
                    allloss=discover_loss+args.w_kd*kdloss
                    
                    allloss.backward()  
                    optimizer.step()
            
        netF_T.eval()
        netB_T.eval()
        netC.eval()
        if args.dset=='VISDA-C':
            acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF_T, netB_T, netC, True)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, args.max_epoch, acc_s_te) + '\n' + acc_list
        


        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str+'\n')
        netF_T.train()
        netB_T.train()
        if args.issave:
            torch.save(netF_T.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename+ str(epoch) + ".pt"))
            torch.save(netB_T.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + str(epoch) +".pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + str(epoch) +".pt"))

        

    if args.issave:
        torch.save(netF_T.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB_T.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF_T, netB_T, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))
def get_mean(f1,f2):
    f1=f1.view(f1.shape[0],f1.shape[1],-1).cpu().detach().numpy()
    f2=f2.view(f2.shape[0],f2.shape[1],-1).cpu().detach().numpy()
    x=f1.shape[0]
    n=f1.shape[1]
    #pdb.set_trace()
    mean=[]
    for nn in range(0,n):
        dis=0
        for i in range(0,x):
            #pdb.set_trace()
            dis+=euclidean(f1[i,nn,:],f2[i,nn,:])
        dis=dis/x
        mean.append(dis)
    #pdb.set_trace()
    mean=torch.from_numpy(np.array(mean)) 
    return mean
def get_means(x1,x2,x3,x11,x21,x31):
    #pdb.set_trace()

    mean1=get_mean(x1,x11)
    mean2=get_mean(x2,x21)
    mean3=get_mean(x3,x31)

    return mean1,mean2,mean3
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def get_list(a,b):
    
    if a.shape != () and b.shape !=  ():
        tmp = [val for val in a if val in b] 
    else: 
        tmp=[]
    return tmp



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--threshold', type=int, default=2.5)
    parser.add_argument('--w_kd', type=int, default=10)
    parser.add_argument('--w_forget', type=int, default=0.9)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/target/visda')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i ==args.s:
            continue
        args.t = i

        folder = './data'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)
        
        copyfile('./six_visda.py', f'./{args.output_dir}/six_visda.py')
        args.savename = 'par_' + str(args.cls_par)
        
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)
