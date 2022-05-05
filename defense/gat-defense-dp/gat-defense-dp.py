"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
# from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gat import GAT,GAT_pia
from utils import EarlyStopping
from data_loader import *

# from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd

import pickle

import math

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def stable_sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,e = model(features)
        logits = logits[mask]
        labels = labels[mask]
        scores, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        out = scores.cpu().detach().numpy()
        y_pred = indices.cpu().detach().numpy()
        y_label = labels.cpu().detach().numpy()
        acc = accuracy_score(y_label, y_pred)
        recall = recall_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred)
        f1 = f1_score(y_label, y_pred)
        auc = roc_auc_score(y_label, out)
        res=[acc, recall, precision, f1, auc]
        return accuracy(logits, labels),logits,labels,res
def evaluate2(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,e = model(features)
        logits = logits[mask]
        labels = labels[mask]
        scores, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)

        out = scores.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()
        logits_ = []
        for i in logits:
            logit = [stable_sigmoid(i[0]) / (stable_sigmoid(i[0]) + stable_sigmoid(i[1]) + stable_sigmoid(i[2])),
                     stable_sigmoid(i[1]) / (stable_sigmoid(i[0]) + stable_sigmoid(i[1]) + stable_sigmoid(i[2])),
                     stable_sigmoid(i[2]) / (stable_sigmoid(i[0]) + stable_sigmoid(i[1]) + stable_sigmoid(i[2]))]
            logits_.append(logit)
        # print(logits_)
        logits_ = np.array(logits_)
        y_pred = indices.cpu().detach().numpy()
        y_label = labels.cpu().detach().numpy()
        acc = accuracy_score(y_label, y_pred)
        recall = recall_score(y_label, y_pred, average='macro')
        precision = precision_score(y_label, y_pred, average='macro')
        f1 = f1_score(y_label, y_pred, average='macro')
        auc = roc_auc_score(y_label, logits_, average='macro', multi_class="ovr")
        res=[acc, recall, precision, f1, auc]

        logits = torch.from_numpy(logits).cuda()
        return accuracy(logits, labels),logits,labels,res



def main(args,data,rnd,result_train,result_test):
    # load and preprocess dataset
    # if args.dataset == 'cora':
    #     data = CoraGraphDataset()
    # elif args.dataset == 'citeseer':
    #     data = CiteseerGraphDataset()
    # elif args.dataset == 'pubmed':
    #     data = PubmedGraphDataset()
    # else:
    #     raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data

    print(g)
    # exit()
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    # print(features)
    # print(type(features))
    labels = g.ndata['label']
    # print(labels)
    # print(type(labels))
    train_mask = g.ndata['train_mask']
    # print(train_mask)
    # print(type(train_mask))
    # exit()
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    test_mask2 = g.ndata['test_mask2']
    num_feats = features.shape[1]
    n_classes = 2
    n_edges = len(g.edata['weight'])
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT_pia(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    # print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.device = torch.device("cuda" if args.gpu >= 0 else "cpu")

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits,embed = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()

        grads = [torch.zeros(p.shape).to(args.device) for p in model.parameters()]
        # print((grads))
        igrad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        # print(igrad)
        # for i in igrad:
        #     print(i.size())

        # exit()

        l2_norm = torch.tensor(0.0).to(args.device)
        for g1 in igrad:
            l2_norm += g1.norm(2) ** 2
            # l2_norm += g.sum().square().tolist()
        # print('time12:', int(time.time() / 1000))
        # l2_norm = l2_norm.sqrt()
        divisor = max(torch.tensor(1.0).to(args.device), l2_norm)
        for i in range(len(igrad)):
            grads[i] += igrad[i] / divisor
            # grads[i] += igrad[i]

        for i in range(len(grads)):
            # print(grads[i])
            grads[i] += sigma * (torch.randn_like(grads[i]).to(args.device))
            # print(grads[i])
            grads[i].detach_()
            # exit()

        p_list = [p for p in model.parameters()]
        for i in range(len(p_list)):
            p_list[i].grad = grads[i]
            # print(p_list[i].grad)
            p_list[i].grad.detach_()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])



        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc,_,_,_ = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        # print()

        if epoch %20==0:


            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                  format(epoch, np.mean(dur), loss.item(), train_acc,
                         val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    # acc = evaluate(model, features, labels, test_mask)


    if args.dataset == 'pubmed' or args.dataset == 'pubmed-interintra':

        acc1, score1, pred1, eval_train = evaluate2(model, features, labels, train_mask)
        acc2, score2, pred2, eval_test = evaluate2(model, features, labels, test_mask2)

    else:

        acc1, score1, pred1, eval_train = evaluate(model, features, labels, train_mask)
        acc2, score2, pred2, eval_test = evaluate(model, features, labels, test_mask2)

    print(ii, sed)
    print("Test Accuracy {:.4f}".format(acc2))

    emb_matrix1 = embed[0].cpu().detach().numpy()
    emb_matrix2 = embed[1].cpu().detach().numpy()

    output_train = score1.cpu().detach().numpy()
    output_test = score2.cpu().detach().numpy()

    result_train.append(eval_train)
    result_test.append(eval_test)

    para= {}
    cnt = 0

    for p in model.parameters():
        # print(p)
        p = p.detach().numpy()
        # print(p)
        para[cnt] = p
        cnt += 1

    # for name, param in model.named_parameters():
    #     print(name,param.size())


    # gat_layers.0.attn_l torch.Size([1, 8, 8])
    # gat_layers.0.attn_r torch.Size([1, 8, 8])
    # gat_layers.0.bias torch.Size([64])
    # gat_layers.0.fc.weight torch.Size([64, 5])
    # gat_layers.1.attn_l torch.Size([1, 8, 8])
    # gat_layers.1.attn_r torch.Size([1, 8, 8])
    # gat_layers.1.bias torch.Size([64])
    # gat_layers.1.fc.weight torch.Size([64, 64])
    # gat_layers.2.attn_l torch.Size([1, 1, 2])
    # gat_layers.2.attn_r torch.Size([1, 1, 2])
    # gat_layers.2.bias torch.Size([2])
    # gat_layers.2.fc.weight torch.Size([2, 64])


    tp='%-dp-%s'%(args.dataset,sigma)

    # with open("./{}/pokec-para-{}-{}-12-13-{}.pkl".format(tp,str(ii), str(sed),tp), "wb") as f:
    #     pickle.dump(para, f)


    with open('./%s/embed1-%s-%s-%s.txt' % (tp,ii, sed,tp), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix1)[0], args.num_hidden))
        for item in emb_matrix1:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/embed2-%s-%s-%s.txt' % (tp,ii, sed,tp), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix2)[0], args.num_hidden))
        for item in emb_matrix2:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()


    with open('./%s/output_train-%s-%s-%s.txt' % (tp,ii, sed,tp), 'w') as f:
        for item in output_train:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/output_test-%s-%s-%s.txt' % (tp,ii,sed,tp), 'w') as f:
        for item in output_test:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()


    # if (rnd + 1) % 100 == 0:
    #     data1 = pd.DataFrame(result_train)
    #     data1.to_csv('./fb-edu-gender/result_train-fb-edu-gender-%s.csv' % (rnd))
    #
    #     data2 = pd.DataFrame(result_test)
    #     data2.to_csv('./fb-edu-gender/result_test-fb-edu-gender-%s.csv' % (rnd))
    #
    # rnd += 1




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--dataset', type=str, default='pokec',
                        help='One of {pokec,fb,pubmed,pokec-interintra,fb-interintra,pubmed-interintra}')

    args = parser.parse_args()
    print(args)

    rnd = 0
    result_train = []
    result_test = []
    global sigma
    sigma=48.448 #noise scale, epsilon=10/5/1/0.5/0.1  sigma=0.48/2.4/4.8/24.224/48.448
    data_dir = '/content/drive/My Drive/pygcn-master2/data/{}/'.format(args.dataset)

    for sed in range(1, 6):
        for ii in range(-100, 100):

            if args.dataset == 'pokec':
                dataset= process_pokec(data_dir,ii, sed)

            elif args.dataset == 'fb':
                dataset = process_fb(data_dir,ii, sed)

            elif args.dataset == 'pubmed':
                dataset = process_pubmed(data_dir,ii, sed)

            elif args.dataset == 'pokec-interintra':
                dataset = process_pokec_interintra(data_dir,ii, sed)

            elif args.dataset == 'fb-interintra':
                dataset = process_fb_interintra(data_dir,ii, sed)

            elif args.dataset == 'pubmed-interintra':
                dataset = process_pubmed_interintra(data_dir,ii, sed)

            main(args,dataset,rnd,result_train,result_test)
