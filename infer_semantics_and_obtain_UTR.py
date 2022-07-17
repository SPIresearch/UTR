import torch
import torch.nn as nn
import numpy as np
import copy
import network
from scipy.spatial.distance import cdist
import random
from collections import Counter
def sa(netF):#sensitivity analysis for measuring uncertainty
    netF_perturbed = network.ResBase(res_name="resnet101").cuda()
    
    netF_perturbed.load_state_dict(netF.state_dict())
    state=netF_perturbed.state_dict()
    for k,v, in state.items():
        #pdb.set_trace()
            a=torch.rand_like(state[k],dtype=torch.float)
            a=(a-0.5)/10+1
            state[k]=torch.mul(state[k],a)
    netF_perturbed.load_state_dict(state)
    #network.load_state_dict(pra)
    return netF_perturbed


def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))

def get_list(a,b):
    
    if a.shape != () and b.shape !=  ():
        tmp = [val for val in a if val in b] 
    else: 
        tmp=[]
    return tmp

def infer_semantics_and_obtain_UTR(loader,netF_T, netB_T, netC,args,k=1000):
    start_test = True
    netF_perturbed=sa(netF_T)

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB_T(netF_T(inputs))
            
            feas1 = netB_T(netF_perturbed(inputs))
            outputs = netC(feas)
            outputs1 = netC(feas1)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_fea1 = feas1.float().cpu()
                all_output1 = outputs1.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_fea1 = torch.cat((all_fea1, feas1.float().cpu()), 0)
                all_output1 = torch.cat((all_output1, outputs1.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                #all_output2 = torch.cat((all_output2, outputs2.float().cpu()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, semantics = torch.max(all_output, 1)
    print('prediction_accuracy:',torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]))
    UTR_matrix=(all_fea-all_fea1)*(all_fea-all_fea1)
    UTR_I=torch.mean(UTR_matrix,dim=1)
    UTR_I=UTR_I/(UTR_I.mean())
    UTR_D=torch.mean(UTR_matrix,dim=0)
    threshold=min(3,UTR_I.max())
    high_risk=torch.nonzero(UTR_I>threshold).squeeze().cpu().numpy()
    high_risk=[i for i in high_risk]
    low_risk=torch.nonzero(UTR_I>threshold).squeeze().cpu().numpy()
    low_risk=[i for i in low_risk]
    data_list=[i for i in range(all_fea.shape[0])]

    all_fea=all_fea[data_list]
    all_output=all_output[data_list]
    predict=predict[data_list]
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>=args.threshold)
    labelset = labelset[0]
    
    #pseudo_label strategy of SHOT
    dd = cdist(all_fea, initc[labelset], args.distance)
    
    

    
    
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
     # if there are no semantics of a certainty class, assign some samples to this class
    cc=Counter(semantics[low_risk].cpu().numpy()) 
    have_semantics=list(cc.keys())
    no_semantics_dict={}
    if len(have_semantics)<12:
        no_semantics=list(set(range(12))-set(have_semantics))
    else:
        no_semantics=list(range(12))
        k=100
    random.shuffle(no_semantics)
    for i in no_semantics:
        sample=np.argsort(dd[:,i])
        sample=sample[:k]
        if i not in no_semantics_dict.keys():
            no_semantics_dict[i]=[]
        for idx, ix in enumerate(sample):
            no_semantics_dict[i].append(data_list[ix])
         
    acc = np.sum(pred_label == all_label[data_list].float().numpy()) / len(all_fea)
    log_str = 'all clust Accuracy ={:.2f}%'.format( acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)
    supplement=[]
   
    for key,val in no_semantics_dict.items():
        for i,j in enumerate(val):
            if j not in supplement:
                supplement.append(j)
              
                semantics[j]=key

    return semantics.long(),UTR_I,UTR_D,high_risk,have_semantics,no_semantics,data_list


