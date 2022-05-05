"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from data_loader import *
from utils import *
import pandas as pd

import pickle

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
class GraphSAGE_pia(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_pia, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        embed=[]
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                embed.append(h)
                h = self.activation(h)
                h = self.dropout(h)
        return h,embed


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits,e = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
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

        return correct.item() * 1.0 / len(labels),logits,labels,res



import math

def stable_sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

def evaluate2(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits,e = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        scores, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        out = scores.cpu().detach().numpy()
        logits=logits.cpu().detach().numpy()
        logits_=[]
        for i in logits:
            logit=[stable_sigmoid(i[0])/(stable_sigmoid(i[0])+stable_sigmoid(i[1])+stable_sigmoid(i[2])),stable_sigmoid(i[1])/(stable_sigmoid(i[0])+stable_sigmoid(i[1])+stable_sigmoid(i[2])),stable_sigmoid(i[2])/(stable_sigmoid(i[0])+stable_sigmoid(i[1])+stable_sigmoid(i[2]))]
            logits_.append(logit)
        # print(logits_)
        logits_=np.array(logits_)
        y_pred = indices.cpu().detach().numpy()
        y_label = labels.cpu().detach().numpy()
        acc = accuracy_score(y_label, y_pred)
        recall = recall_score(y_label, y_pred,average='macro')
        precision = precision_score(y_label, y_pred,average='macro')
        f1 = f1_score(y_label, y_pred,average='macro')
        auc = roc_auc_score(y_label, logits_,average='macro',multi_class="ovr")
        res=[acc, recall, precision, f1, auc]
        logits=torch.from_numpy(logits)

        return correct.item() * 1.0 / len(labels),logits,labels,res


def main(args,data,rnd,result_train,result_test):
    # load and preprocess dataset
    # data = load_data(args)
    # g = data[0]

    g = data
    features = g.ndata['feat']
    print(np.shape(features))
    labels = g.ndata['label']
    print(np.shape(labels))
    print(labels.int().sum().item())
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    test_mask2 = g.ndata['test_mask2']
    in_feats = features.shape[1]
    # n_classes = data.num_classes
    # n_edges = data.graph.number_of_edges()
    num_feats = features.shape[1]
    n_classes = 2
    n_edges=len(g.edata['weight'])
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

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    test_nid2 = test_mask2.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()

    if args.early_stop:
        stopper = EarlyStopping(patience=100)

    if cuda:
        g = g.int().to(args.gpu)

    # create GraphSAGE model
    model = GraphSAGE_pia(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits,embed = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)



        acc,_,_,_ = evaluate(model, g, features, labels, val_nid)

        if epoch % 20 == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                acc, n_edges / np.mean(dur) / 1000))

        if args.early_stop:
            if stopper.step(acc, model):
                break

    # print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    # acc = evaluate(model, features, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(acc))

    if args.dataset == 'pubmed' or args.dataset == 'pubmed-interintra':
        acc1, score1, pred1, eval_train = evaluate2(model, g,features, labels, train_mask)
        acc2, score2, pred2, eval_test = evaluate2(model, g,features, labels, test_mask2)

    else:
        acc1, score1, pred1, eval_train = evaluate(model, g, features, labels, train_nid)
        acc2,score2,pred2,eval_test = evaluate(model, g, features, labels, test_nid2)


    print(ii,sed)
    print("Test Accuracy {:.4f}".format(acc2))


    emb_matrix1 = embed[0].cpu().detach().numpy()
    emb_matrix2 = embed[1].cpu().detach().numpy()

    output_train = score1.cpu().detach().numpy()
    output_test = score2.cpu().detach().numpy()

    result_train.append(eval_train)
    result_test.append(eval_test)

    para= {}
    cnt = 0

    # for p in model.parameters():
    #     # print(p)
    #     p = p.detach().numpy()
    #     # print(p)
    #     para[cnt] = p
    #     cnt += 1

    # print(embed,emb_matrix1,emb_matrix2)
    # print(np.shape(emb_matrix1), np.shape(emb_matrix2))

    # for name, param in model.named_parameters():
    #     print(name,param.size())

    # layers.0.bias torch.Size([16])
    # layers.0.fc_neigh.weight torch.Size([16, 5])
    # layers.1.bias torch.Size([16])
    # layers.1.fc_neigh.weight torch.Size([16, 16])
    # layers.2.bias torch.Size([2])
    # layers.2.fc_neigh.weigh torch.Size([2, 16])
    tp = args.dataset

    # with open("./{}/pokec-para-{}-{}-12-13-{}.pkl".format(tp,str(ii), str(sed),tp), "wb") as f:
    #     pickle.dump(para, f)


    with open('./%s/embed1-%s-%s-%s.txt' % (tp,ii, sed,tp), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix1)[0], args.n_hidden))
        for item in emb_matrix1:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/embed2-%s-%s-%s.txt' % (tp,ii, sed,tp), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix2)[0], args.n_hidden))
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
        for item in output_train:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()
        for item in output_test:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()


    # rnd += 1
    #
    # if (rnd + 1) % 100 == 0:
    #     data1 = pd.DataFrame(result_train)
    #     data1.to_csv('./fb-edu-gender/result_train-fb-edu-gender-%s.csv' % (rnd))
    #
    #     data2 = pd.DataFrame(result_test)
    #     data2.to_csv('./fb-edu-gender/result_test-fb-edu-gender-%s.csv' % (rnd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")

    parser.add_argument('--dataset', type=str, default='pokec',
                        help='One of {pokec,fb,pubmed,pokec-interintra,fb-interintra,pubmed-interintra}')
    args = parser.parse_args()
    print(args)

    rnd = 0
    result_train = []
    result_test = []
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

