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

import loss
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

import network
from data_list import ImageList, ImageList_idx, Listset


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


from randaug import RandAugmentMC


def aug_mix(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):

    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(
        txt_tar, transform=image_train(), transform1=aug_mix()
    )
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders, dsets


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def capture_unc(pre, network1):

    network1.load_state_dict(pre)
    pre1 = network1.state_dict()

    for (
        k,
        v,
    ) in pre1.items():

        a = torch.rand_like(pre1[k], dtype=torch.float)
        a = (a - 0.5) / 10 + 1
        pre1[k] = torch.mul(pre1[k], a)
    network1.load_state_dict(pre1)

    return network1


from loss import CrossEntropy1, CrossEntropyLabelSmooth


def train_target(args):
    dset_loaders, dsets = data_load(args)
    writer = SummaryWriter(comment=f"image_mttwoset_kl")

    if args.net[0:3] == "res":
        netF = network.ResBase(res_name=args.net).cuda()
        netF1 = network.ResBase(res_name=args.net).cuda()
        netFS = network.ResBase(res_name=args.net).cuda()
        netFS1 = network.ResBase(res_name=args.net).cuda()
        netF_f = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == "vgg":
        netF = network.VGGBase(vgg_name=args.net).cuda()
        netF1 = network.VGGBase(vgg_name=args.net).cuda()
    netB_f = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    netB1 = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC1 = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    netBS = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netCS = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    netBS1 = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netCS1 = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    modelpath = args.output_dir_src + "/source_F.pt"

    netF.load_state_dict(torch.load(modelpath))
    netF1.load_state_dict(torch.load(modelpath))
    netFS.load_state_dict(torch.load(modelpath))
    netFS1.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + "/source_B.pt"
    netB.load_state_dict(torch.load(modelpath))
    netB1.load_state_dict(torch.load(modelpath))
    netBS.load_state_dict(torch.load(modelpath))
    netBS1.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + "/source_C.pt"
    netC.load_state_dict(torch.load(modelpath))
    netC1.load_state_dict(torch.load(modelpath))
    netCS.load_state_dict(torch.load(modelpath))
    netCS1.load_state_dict(torch.load(modelpath))

    netC.eval()
    netC1.eval()
    netB1.eval()
    netF1.eval()
    netBS.eval()
    netFS.eval()
    netBS1.eval()
    netFS1.eval()

    param_group = []

    for k, v in netF.named_parameters():
        param_group += [{"params": v, "lr": args.lr * 0.1}]

    for k, v in netB.named_parameters():
        param_group += [{"params": v, "lr": args.lr}]

    optimizer = optim.SGD(param_group, weight_decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = op_copy(optimizer)

    print(len(dsets["target"]))

    iter_num = 0

    start = True
    epoch = 0

    while epoch < 8:
        epoch += 1
        iter_num = 0
        lr_scheduler(optimizer, iter_num=epoch, max_iter=args.max_epoch)
        while iter_num * args.batch_size < 55388:
            try:
                [inputs_test, inputs_testa], label, tar_idx = iter_test.next()
            except:
                all_list = range(55388)

                dset_all = Listset(dsets["target"], all_list)
                dset_all = DataLoader(
                    dset_all, batch_size=args.batch_size, shuffle=True, drop_last=True
                )
                iter_test = iter(dset_all)
                [inputs_test, inputs_testa], label, tar_idx = iter_test.next()

            if inputs_test.size(0) == 1:
                continue

            inputs_test = inputs_test.cuda()
            iter_num += 1

            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(
                        -msoftmax * torch.log(msoftmax + args.epsilon)
                    )
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * args.ent_par

            optimizer.zero_grad()
            im_loss.backward()
            optimizer.step()

    while epoch < args.max_epoch:
        netF.eval()
        netB.eval()
        netC.eval()
        netF1.eval()
        netB1.eval()
        netC1.eval()
        lr_scheduler(optimizer, iter_num=epoch, max_iter=args.max_epoch)
        pre = copy.deepcopy(netF.state_dict())
        netF1 = capture_unc(pre, netF1)

        mem_label, mix_set1, mix_set2, high_risk, least_num_per_class_idx, all_list = (
            infer_semantics_and_obtain_UTR(
                dset_loaders["test"], netF, netB, netC, netF1, netB1, netC1, args
            )
        )

        netF1.load_state_dict(netF.state_dict())

        mem_label = mem_label.cuda()

        netF.train()
        netB.train()
        netC.train()

        iter_hphc_num1 = 0

        mix_set1 = list(set(mix_set1) | set(least_num_per_class_idx))

        dset_all = Listset(dsets["target"], all_list)
        dset_all = DataLoader(
            dset_all, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        iter_num = 0
        while (iter_num * (args.batch_size // 4)) < max(len(mix_set2), len(mix_set1)):
            iter_num += 1

            if len(mix_set1) != 0 and len(mix_set2) != 0:
                try:
                    [inputs_test, inputs_testa], label, tar_idx = iter_set1.next()
                except:
                    dset_mix_set1 = Listset(dsets["target"], mix_set1)
                    dloader_mix_set1 = DataLoader(
                        dset_mix_set1,
                        batch_size=args.batch_size // 4,
                        shuffle=True,
                        drop_last=True,
                    )
                    iter_set1 = iter(dloader_mix_set1)
                    [inputs_test, inputs_testa], label, tar_idx = iter_set1.next()
                try:
                    [inputs_testu, inputs_testua], labelu, tar_idxu = iter_set2.next()
                except:
                    dset_mix_set2 = Listset(dsets["target"], mix_set2)
                    dloader_mix_set2 = DataLoader(
                        dset_mix_set2,
                        batch_size=args.batch_size // 4,
                        shuffle=True,
                        drop_last=True,
                    )
                    iter_set2 = iter(dloader_mix_set2)
                    [inputs_testu, inputs_testua], labelu, tar_idxu = iter_set2.next()

                if inputs_test.size(0) > 1 and inputs_testu.size(0) > 1:

                    optimizer.zero_grad()

                    inputs_test = inputs_test.cuda()

                    inputs_testu = inputs_testu.cuda()
                    inputs_testua = inputs_testua.cuda()

                    with torch.no_grad():
                        pred_u = netC(netB(netF(inputs_testu)))
                        pred_u = nn.Softmax(dim=1)(pred_u)

                        pred_u = pred_u ** (1 / 0.5)
                        targets_u = pred_u / pred_u.sum(dim=1, keepdim=True)
                        targets_u = targets_u.detach()

                    targets_x = mem_label[tar_idx]
                    targets_x = (
                        torch.zeros((targets_x.size(0), 12))
                        .scatter_(1, targets_x.unsqueeze(1).cpu(), 1)
                        .cuda()
                    )

                    all_inputs = torch.cat(
                        [inputs_test, inputs_testu, inputs_testua], dim=0
                    )

                    all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
                    args.alpha = 0.75
                    l = np.random.beta(args.alpha, args.alpha)

                    l = max(l, 1 - l)

                    idx = torch.randperm(all_inputs.size(0))

                    input_a, input_b = all_inputs, all_inputs[idx]
                    target_a, target_b = all_targets, all_targets[idx]

                    mixed_input = l * input_a + (1 - l) * input_b
                    mixed_target = l * target_a + (1 - l) * target_b

                    mixed_input = list(torch.split(mixed_input, args.batch_size // 4))

                    mixed_input = interleave(mixed_input, args.batch_size // 4)
                    logits = [netC(netB(netF(mixed_input[0])))]
                    for input in mixed_input[1:]:
                        logits.append(netC(netB(netF(input))))
                    logits = interleave(logits, args.batch_size // 4)
                    logits_x = logits[0]
                    logits_u = torch.cat(logits[1:], dim=0)

                    if (
                        args.cls_par > 0
                        and logits_u.shape[0] == 32
                        and logits_x.shape[0] == 16
                    ):
                        classifier_loss = CrossEntropy1()(
                            logits_x, mixed_target[: args.batch_size // 4]
                        )
                        classifier_loss += CrossEntropy1()(
                            logits_u, mixed_target[args.batch_size // 4 :]
                        )

                        hplcloss = classifier_loss

                        optimizer.zero_grad()
                        hplcloss.backward()

                        optimizer.step()

        pre = copy.deepcopy(netFS.state_dict())
        netFS1 = capture_unc(pre, netFS1)
        pre = copy.deepcopy(netF.state_dict())
        netF1 = capture_unc(pre, netF1)
        pre = copy.deepcopy(netB.state_dict())
        netB1 = capture_unc(pre, netB1)

        iter_num = 0
        while iter_hphc_num1 * (args.batch_size // 4) < len(
            high_risk
        ) or iter_num * args.batch_size < len(all_list):
            iter_num += 1

            if len(high_risk) != 0 and iter_hphc_num1 * (args.batch_size // 4) < len(
                high_risk
            ):
                try:
                    [inputs_test, inputs_testa], label, tar_idxhr = next(iter_high_risk)
                except:
                    dset_high_risk = Listset(dsets["target"], high_risk)
                    dloader_high_risk = DataLoader(
                        dset_high_risk,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=False,
                    )
                    iter_high_risk = iter((dloader_high_risk))
                    [inputs_test, inputs_testa], label, tar_idxhr = next(iter_high_risk)
                iter_hphc_num1 += 1
                if inputs_test.size(0) > 1:
                    pred = mem_label[tar_idxhr]

                    valid = torch.nonzero(pred != -1).squeeze()

                    pred = pred[valid]
                    if valid.shape[0] <= 1:
                        continue
                    optimizer.zero_grad()

                    inputs_test = inputs_test[valid].cuda()

                    features_test = netB(netF(inputs_test))
                    outputs_test = netC(features_test)

                    if args.cls_par > 0:
                        classifier_loss = 0.9 * nn.CrossEntropyLoss()(
                            outputs_test, pred
                        )
                    else:
                        classifier_loss = torch.tensor(0.0).cuda()

                    hphcloss = -classifier_loss

                    optimizer.zero_grad()
                    hphcloss.backward()

                    optimizer.step()

            if len(all_list) != 0 and iter_num * args.batch_size < len(all_list):
                try:
                    [inputs_test, inputs_testa], label, tar_idxall = next(iter_all)
                except:
                    dset_all = Listset(dsets["target"], all_list)
                    dset_all = DataLoader(
                        dset_all,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                    iter_all = iter((dset_all))
                    [inputs_test, inputs_testa], label, tar_idxall = next(iter_all)

                if inputs_test.size(0) > 1:

                    optimizer.zero_grad()

                    pred = mem_label[tar_idxall]

                    valid = torch.nonzero(pred != -2).squeeze()

                    if valid.shape[0] <= 1:
                        continue
                    inputs_test = inputs_test[valid].cuda()

                    inputs_testa = inputs_testa[valid].cuda()
                    pred = pred[valid].cuda()
                    features_test = netB(netF(inputs_test))
                    outputs_test = netC(features_test)

                    with torch.no_grad():

                        features_testS1 = netBS1(netFS1(inputs_test))
                        features_testS = netBS(netFS(inputs_test))
                        UTR_D = 1 / 4 * get_mean(features_testS1, features_testS).cuda()
                    kdloss = torch.nn.MSELoss(reduce=False, size_average=False)(
                        features_test, features_testS
                    )

                    QUTR_D = torch.sigmoid(-UTR_D)
                    kdloss = torch.mean(kdloss * QUTR_D)

                    if args.ent:
                        softmax_out = nn.Softmax(dim=1)(outputs_test)
                        entropy_loss = torch.mean(loss.Entropy(softmax_out))
                        if args.gent:
                            msoftmax = softmax_out.mean(dim=0)
                            entropy_loss -= torch.sum(
                                -msoftmax * torch.log(msoftmax + 1e-5)
                            )

                        im_loss = entropy_loss * args.ent_par

                    if epoch <= 10:
                        x = 10
                    else:
                        x = 0
                    allloss = im_loss + x * kdloss

                    allloss.backward()
                    optimizer.step()

        epoch += 2

        netF.eval()
        netB.eval()
        netC.eval()
        if args.dset == "VISDA-C":
            acc_s_te, acc_list = cal_acc(dset_loaders["test"], netF, netB, netC, True)
            log_str = (
                "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
                    args.name, epoch, args.max_epoch, acc_s_te
                )
                + "\n"
                + acc_list
            )
        else:

            acc_s_te, _ = cal_acc(dset_loaders["test"], netF, netB, netC, False)
            log_str = "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
                args.name, epoch, args.max_epoch, acc_s_te
            )

        args.out_file.write(log_str + "\n")
        args.out_file.flush()
        print(log_str + "\n")
        netF.train()
        netB.train()
        if args.issave:
            torch.save(
                netF.state_dict(),
                osp.join(args.output_dir, "target_F_" + str(epoch) + ".pt"),
            )
            torch.save(
                netB.state_dict(),
                osp.join(args.output_dir, "target_B_" + str(epoch) + ".pt"),
            )
            torch.save(
                netC.state_dict(),
                osp.join(args.output_dir, "target_C_" + str(epoch) + ".pt"),
            )

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def get_mean(f1, f2):
    f1 = f1.view(f1.shape[0], f1.shape[1], -1).cpu().detach().numpy()
    f2 = f2.view(f2.shape[0], f2.shape[1], -1).cpu().detach().numpy()
    x = f1.shape[0]
    n = f1.shape[1]

    mean = []
    for nn in range(0, n):
        dis = 0
        for i in range(0, x):

            dis += euclidean(f1[i, nn, :], f2[i, nn, :])
        dis = dis / x
        mean.append(dis)

    mean = torch.from_numpy(np.array(mean))
    return mean


def get_means(x1, x2, x3, x11, x21, x31):

    mean1 = get_mean(x1, x11)
    mean2 = get_mean(x2, x21)
    mean3 = get_mean(x3, x31)

    return mean1, mean2, mean3


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
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def get_list(a, b):

    if a.shape != () and b.shape != ():
        tmp = [val for val in a if val in b]
    else:
        tmp = []
    return tmp


def infer_semantics_and_obtain_UTR(loader, netF, netB, netC, netF1, netB1, netC1, args):
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))

            feas1 = netB1(netF1(inputs))
            outputs = netC(feas)
            outputs1 = netC1(feas1)
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

    all_output = nn.Softmax(dim=1)(all_output)
    preval, predict = torch.max(all_output, 1)
    print(
        "direct_all_accuracy:",
        torch.sum(torch.squeeze(predict).float() == all_label).item()
        / float(all_label.size()[0]),
    )

    UTR_I = torch.sum((all_fea - all_fea1) * (all_fea - all_fea1), dim=1)

    tc = min((UTR_I.mean()) * 3, UTR_I.max())

    high_risk = torch.nonzero(UTR_I > tc).squeeze().cpu().numpy()
    low_risk = torch.nonzero(UTR_I <= tc).squeeze().cpu().numpy()

    set1 = torch.nonzero(preval > 0.9).squeeze().cpu().numpy()
    set2 = torch.nonzero(preval <= 0.9).squeeze().cpu().numpy()
    mix_set1 = get_list(set1, low_risk)

    mix_set2 = get_list(set2, low_risk)
    all_list = [i for i in range(all_fea.shape[0])]

    _, predict = torch.max(all_output, 1)

    print("UTR_I:", UTR_I.mean())

    high_risk = get_list(high_risk, high_risk)

    mix_set1 = mix_set1
    mix_set1_label = predict[mix_set1]

    least_num_per_class = 100
    semi_unlabeled_idx = list(set(set1) - set(mix_set1) - set(high_risk))
    clu_sample = all_list

    final_label = predict
    all_fea = all_fea[clu_sample]
    all_output = all_output[clu_sample]
    predict = predict[clu_sample]
    if args.distance == "cosine":
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count >= args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)

    least_num_dict = {}

    classes_set = list(range(12))
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    random.shuffle(classes_set)
    for i in classes_set:

        sample = np.argsort(dd[:, i])

        sample = sample[:least_num_per_class]

        if i not in least_num_dict.keys():
            least_num_dict[i] = []

        for idx, ix in enumerate(sample):
            least_num_dict[i].append(clu_sample[ix])

    acc = np.sum(pred_label == all_label[clu_sample].float().numpy()) / len(all_fea)
    log_str = "clust Accuracy ={:.2f}%".format(acc * 100)

    args.out_file.write(log_str + "\n")
    args.out_file.flush()
    print(log_str)

    least_num_per_class_idx = []
    for i, j in enumerate(mix_set1):
        final_label[j] = mix_set1_label[i]

    for key, val in least_num_dict.items():
        for i, j in enumerate(val):
            if j not in least_num_per_class_idx:
                least_num_per_class_idx.append(j)

                final_label[j] = key

    final_label = final_label.cuda()

    return (
        final_label.long(),
        mix_set1,
        mix_set2,
        least_num_per_class_idx,
        high_risk,
        all_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHOT")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=40, help="max iterations")
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dset",
        type=str,
        default="VISDA-C",
        choices=["VISDA-C", "office", "office-home", "office-caltech"],
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--net", type=str, default="resnet101", help="alexnet, vgg16, resnet50, res101"
    )
    parser.add_argument("--seed", type=int, default=2020, help="random seed")

    parser.add_argument("--gent", type=bool, default=True)
    parser.add_argument("--ent", type=bool, default=True)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--cls_par", type=float, default=0.3)
    parser.add_argument("--ent_par", type=float, default=1.0)
    parser.add_argument("--lr_decay1", type=float, default=0.1)
    parser.add_argument("--lr_decay2", type=float, default=1.0)

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument(
        "--distance", type=str, default="cosine", choices=["euclidean", "cosine"]
    )
    parser.add_argument("--output", type=str, default="ckps/target/mixvis")
    parser.add_argument("--output_src", type=str, default="ckps")

    parser.add_argument("--issave", type=bool, default=True)
    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dset == "office":
        names = ["amazon", "dslr", "webcam"]
        args.class_num = 31
    if args.dset == "VISDA-C":
        names = ["train", "validation"]
        args.class_num = 12
    if args.dset == "office-caltech":
        names = ["amazon", "caltech", "dslr", "webcam"]
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = "/home/spi/peijiangbo/"
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src, args.dset, names[args.s][0].upper()
        )
        args.output_dir = osp.join(
            args.output, args.dset, names[args.s][0].upper() + names[args.t][0].upper()
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)
        from shutil import copyfile

        args.savename = "par_" + str(args.cls_par)

        args.out_file = open(
            osp.join(args.output_dir, "log_" + args.savename + ".txt"), "w"
        )
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)
