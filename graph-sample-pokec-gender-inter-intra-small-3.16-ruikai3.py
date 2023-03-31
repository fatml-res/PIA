import networkx as nx
import numpy as np
import pickle as pk
import os
# import pandas as pd
import sys
# import gurobipy as gp
# from gurobipy import GRB
# from new_sampling_methods import *
# from edge_sampling import *
# from base_sampling_methods import *
# from dynamic_sampling import *
# from pygcn.preprocessing import *
import math
import random
import pickle
import pickle as pkl
import networkx as nx
import copy


# from .utils import load_data_pokec_ruikai, accuracy

sys.setrecursionlimit(1000000)

def sample(ft,adj_sparse,pos_sampled_index1,neg_sampled_index1, pos_index,neg_index):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    sample_len=len(pos_sampled_index1)+len(neg_sampled_index1)
    adj=np.zeros((sample_len,sample_len))
    # print(len(graph_sampled_index1))
    index=[]
    for i in pos_sampled_index1:
        index.append(pos_index[i])

    for i in neg_sampled_index1:
        index.append(neg_index[i])

    for i in range(len(index)):
        for j in range(i + 1, len(index)):
            if adj_sparse[index[i]][index[j]] == 1:
                adj[i][j]=1
    ft_list=ft[index]
    return ft_list,adj

# train_edges=[8,9,5,6,5345]
# train_lables=[0,0,1,1,1]
# all = list(range(5))
#
# sp_idx = random.sample(all, 5)
# print(sp_idx)
#
# train_edges1=np.array(train_edges)[np.array(sp_idx)]
# train_labels1=np.array(train_lables)[np.array(sp_idx)]
# print(train_edges1,train_labels1)
# #
# train_lables=[1] * 3 + [0] * 3
# test_lables = [1] * 4+ [0] * 5
# print(train_lables)
# exit()

#
#
# for sed in range(1,2):
#     for ii in range(-100,100):
#         seed=sed
#         # Load data
#         if ii < 0:
#
#             f2 = open("../data/fb-interintra-small-0/fb-adj-feat-{0}-{1}.pkl".format(str(ii), str(sed)), 'rb')
#
#         else:
#             f2 = open("../data/fb-interintra-small-1/fb-adj-feat-{0}-{1}.pkl".format(str(ii), str(sed)), 'rb')
#         adj, ft = pkl.load(f2, encoding='latin1')
#
#         g = nx.Graph(np.array(adj.todense()))
#
#         gender_index = 77
#         edu_index = 53
#
#         feat = ft
#
#         # feat=[]
#
#         print(set(ft[:, gender_index]))
#
#         for i, n in enumerate(g.nodes()):
#             ginfo = ft[n][gender_index]
#             einfo = ft[n][edu_index]
#
#             # print(info)
#             g.nodes[n]['gender'] = ginfo
#             g.nodes[n]['education'] = einfo
#
#         mm = 0
#         mf = 0
#         ff = 0
#
#         for ed in g.edges():
#             e1 = ed[0]
#             e2 = ed[1]
#             if g.nodes[e1]['gender'] + g.nodes[e2]['gender'] == 2:
#                 mm += 1
#             elif g.nodes[e1]['gender'] + g.nodes[e2]['gender'] == 3:
#                 mf += 1
#             elif g.nodes[e1]['gender'] + g.nodes[e2]['gender'] == 4:
#                 ff += 1
#         print(mm, mf, ff, 2 * mf / (mm + ff))
#
#
#         print(ii,2*mf-mm-ff)
#
# exit()

seds=[1,2,3,4,5]
#METHOD = 'bi-deepwalk2'
DATASET = 'facebook-data-new-2'
# DATASET='google+-raw-data-3'
# DATASET='dblp-data'
# fnames = os.listdir('E:\\python\\banlance\\code\\facebook-data-new-2\\')
# fnames = os.listdir('/Users/xiulingwang/Downloads/facebook-data/data/')
# fnames = os.listdir('E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/')
# fnames = os.listdir('E:/python/banlance/code/dblp-data/data/')
# fnames = sorted(fnames)
# print(fnames)

# G_EGO_USERS = set([])
# for fname in fnames:
#     if (fname == '.DS_Store'):
#         continue
#     ustring = fname.split('.')[0]
#     uid = eval(ustring.split('-')[0])
#     # uid = eval(ustring.split('-')[1])
#     if uid in G_EGO_USERS:
#         continue
#     else:
#         G_EGO_USERS.add(uid)
# print(len(G_EGO_USERS))

G_EGO_USERS=['pokec']

rat=1.5

for sed in range(1,6):
    # METHOD = 'bi-deepwalk%s' % (sed)
    sam_idx = 0
    for ego_user in G_EGO_USERS:
        # fnames = os.listdir('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/result/')
        # fnames = sorted(fnames)
        # n = 'test' + str(ego_user) + '_max.csv'
        # print(n)
        # if n in fnames:
        #     continue
        # featurename
        #featname_dir = 'E:/python/banlance/code/facebook-raw-data/' + str(ego_user) + '.featnames'
        # edges
        #edges_dir = 'E:/python/banlance/code/facebook-raw-data/' + str(ego_user) + '.edges'
        # feature
        # feat_dir = './' + str(ego_user) + '-adj-feat.pkl'
        # feat_dir = './' + str(ego_user) + '-adj-feat.pkl'
        # feat_dir = 'E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/' + str(ego_user) + '-adj-feat.pkl'
        # feat_dir = 'E:/python/banlance/code/dblp-data/data/' + str(ego_user) + '-adj-feat.pkl'
        # feat_dir = './pokec-adj-feat-processed-45036.pkl'

        feat_dir = './pokec-adj-feat-processed-45036.pkl'
        f2 = open(feat_dir, 'rb')

        adj, ft = pk.load(f2, encoding='latin1')

        adj0 = np.array(adj.todense(), dtype=np.int32)

        g = nx.Graph(adj)

        gender_index=1
        # edu_index = 53

        feat=ft

        # feat=[]

        print(set(ft[:,gender_index]))

        for i, n in enumerate(g.nodes()):
            ginfo = ft[n][gender_index]
            # einfo = ft[n][edu_index]

            # print(info)
            g.nodes[n]['gender'] = ginfo
            # g.nodes[n]['education'] = einfo

        mm = 0
        mf = 0
        ff = 0

        for ed in g.edges():
            e1 = ed[0]
            e2 = ed[1]
            if g.nodes[e1]['gender'] + g.nodes[e2]['gender'] == 0:
                mm += 1
            elif g.nodes[e1]['gender'] + g.nodes[e2]['gender'] == 1:
                mf += 1
            elif g.nodes[e1]['gender'] + g.nodes[e2]['gender'] == 2:
                ff += 1
        print(ego_user, mm, mf, ff, mf / (mm + ff))


            # exit()

        np.random.seed(sed)  # make sure train-test split is consistent between notebooks
        # adj_sparse = nx.to_scipy_sparse_matrix(g)
        # print(adj_sparse.shape)
        # print(adj_sparse)
        num_nodes=g.number_of_nodes()

        pos_index=[]
        neg_index = []
        for nd in range(g.number_of_nodes()):
            if ft[nd][gender_index]==1:
                pos_index.append(nd)
            else:
                neg_index.append(nd)

        print(len(pos_index),len(neg_index))


        # print(len(pos_index),len(neg_index))

        # exit()
        # sample_cnt=int(1*min(len(pos_index),len(neg_index)))
        sample_cnt = 5000
        idx_pos = list(range(len(pos_index)))
        idx_neg = list(range(len(neg_index)))
        neg_sampled_index1 = random.sample(list(idx_neg), sample_cnt)
        pos_sampled_index1 = random.sample(list(idx_pos), sample_cnt)

        ft_list1, adj1 = sample(feat, adj0, pos_sampled_index1, neg_sampled_index1, idx_pos, idx_neg)

        g1 = nx.Graph(adj1)
        # g1.add_edges_from(edge_list1)
        # g2.add_edges_from(edge_list2)

        adj1 = nx.adjacency_matrix(g1)
        #


        ee = []
        uu = []
        eu = []
        for eg in g1.edges():
            nd1=eg[0]
            nd2=eg[1]
            e1=ft_list1[nd1][gender_index]
            e2=ft_list1[nd2][gender_index]
            if e1 + e2 == 0:
                # print(e1,e2)
                uu.append([eg[0], eg[1]])
            elif e1 + e2 == 1:
                eu.append([eg[0], eg[1]])

            else:
                ee.append([eg[0], eg[1]])


        print(np.shape(uu),np.shape(eu),np.shape(ee))

        sample_cnt=min((np.shape(uu)[0]+np.shape(ee)[0],np.shape(eu)[0]))

        if (np.shape(uu)[0] + np.shape(ee)[0]) > np.shape(eu)[0]:
            exit()

        print(((np.shape(uu)[0] + np.shape(ee)[0]) - np.shape(eu)[0]) / (rat*np.shape(eu)[0]))

        num_nodes_ =g1.number_of_nodes()

        idx_edges = []

        for eg in uu:
            nd0 = eg[0]
            nd1 = eg[1]

            # if ft_list1[nd0][edu_index]==1 or ft_list1[nd1][edu_index]==1:

            if (nd0 > nd1):
                tmp = nd0
                nd0 = nd1
                nd1 = tmp

            edge_idx = nd0 * num_nodes_ + nd1
            idx_edges.append(edge_idx)

        u_nodes = []

        for no in g1.nodes:
            if ft_list1[no][gender_index] == 0:
                u_nodes.append(no)

        u_edge_idx = []
        for i in range(len(u_nodes)):
            for j in range(i + 1, len(u_nodes)):
                u_edge_idx.append(u_nodes[i] * num_nodes_ + u_nodes[j])

        idx_edges = np.array(idx_edges)

        idx_neg_edges = np.setdiff1d(u_edge_idx, idx_edges)

        for ii in range(-100, 100):

            g_orig = copy.deepcopy(g1)

            # g_orig = g1

            step=int((rat*np.shape(eu)[0]-(np.shape(uu)[0]+np.shape(ee)[0]))/100)+1

            num_add=sam_idx*step

            sam_idx += 1

            idx_add_edges = np.array(random.sample(list(idx_neg_edges), num_add))

            add_edges = []

            # for ed in idx_add_edges:
            #     nd1 = int(ed / num_nodes_)
            #     nd2 = ed % num_nodes_
            #     add_edges.append([nd1, nd2])
            #     add_edges.append([nd2, nd1])
            #     g_orig.add_edges_from(add_edges)
            #     print(ft_list1[nd1][1],ft_list1[nd1][2])

            nd1=np.array(idx_add_edges / num_nodes_,dtype='int16')
            nd2=np.array(idx_add_edges % num_nodes_,dtype='int16')
            add_edges=np.concatenate((nd1.reshape(-1,1),nd2.reshape(-1,1)),axis=1)
            g_orig.add_edges_from(list(add_edges))

            tuple1=(nx.adjacency_matrix(g_orig),ft_list1)

            g_ = nx.Graph(g_orig)

            gender_index = 1
            # edu_index = 53

            feat = ft

            # feat=[]

            print(set(ft[:, gender_index]))

            for i, n in enumerate(g_.nodes()):
                ginfo = ft_list1[n][gender_index]
                # einfo = ft_list1[n][edu_index]

                # print(info)
                g_.nodes[n]['gender'] = ginfo
                # g_.nodes[n]['education'] = einfo

            mm = 0
            mf = 0
            ff = 0

            for ed in g_.edges():
                e1 = ed[0]
                e2 = ed[1]
                if g_.nodes[e1]['gender'] + g_.nodes[e2]['gender'] == 0:
                    mm += 1
                elif g_.nodes[e1]['gender'] + g_.nodes[e2]['gender'] == 1:
                    mf += 1
                elif g_.nodes[e1]['gender'] + g_.nodes[e2]['gender'] == 2:
                    ff += 1
            print('***##', ii,mm, mf, ff, 1.5*mf / (mm + ff ))

            # if ii < 0:
            #
            #     with open("/content/drive/My Drive/pygcn-master2/data/fb-interintra-small-0/{0}-adj-feat-{1}-{2}.pkl".format(ego_user,ii,str(sed)), "wb") as f:
            #         pickle.dump(tuple1, f)
            #
            # else:
            #
            #     with open("/content/drive/My Drive/pygcn-master2/data/fb-interintra-small-1/{0}-adj-feat-{1}-{2}.pkl".format(ego_user,ii,str(sed)), "wb") as f:
            #         pickle.dump(tuple1, f)

            if ii < 0:

                with open(
                        "./data/pokec-interintra-small-0-ratio-{3}/{0}-adj-feat-{1}-{2}.pkl".format(
                                ego_user, ii, str(sed),rat), "wb") as f:
                    pickle.dump(tuple1, f)

            else:

                with open(
                        "./data/pokec-interintra-small-1-ratio-{3}/{0}-adj-feat-{1}-{2}.pkl".format(
                                ego_user, ii, str(sed),rat), "wb") as f:
                    pickle.dump(tuple1, f)


















                            #
        # num_sample=800
        #
        # pos_all=list(range(len(pos_index)))
        # neg_all = list(range(len(neg_index)))
        # #
        # pos_len=len(pos_all)
        # neg_len = len(neg_all)
        # #
        # step=int((min(pos_len,neg_len)-800)/100)
        # print(step,pos_len,neg_len)
        # # if step>=50:
        # #     step=50
        #
        # for ii in range(-100,100):
        #     num_sample1=800+ii*step
        #     num_sample2 = 800-ii *step
        #     random.seed(sed)
        #     pos_sampled_index1 = random.sample(pos_all, num_sample1)
        #     neg_sampled_index1 = random.sample(neg_all, num_sample2)
        #
        #     # sampled_index=np.concatenate(( pos_sampled_index1 , neg_sampled_index1))
        #     # print(len(sampled_index))
        #
        #     ft_list1, adj1 = sample(feat, adj0, pos_sampled_index1,neg_sampled_index1, pos_index,neg_index)
        #     # print(ft_list1)
        #     # print(np.shape(ft_list1))
        #
        #     g1 = nx.Graph(adj1)
        #     # g1.add_edges_from(edge_list1)
        #     # g2.add_edges_from(edge_list2)
        #
        #     adj1 = nx.adjacency_matrix(g1)
        #     #
        #     # adj1_dense = np.array(adj1.todense())
        #
        #     tuple1 = (adj1, ft_list1)
        #
        #     # if i<0:
        #     #     label.append(0)
        #     # else:
        #     #     label.append(1)
        #
        #     with open("../data/fb-edu-gender/{0}-adj-feat-{1}-{2}-edu-gender.pkl".format(ego_user,str(ii),str(sed)), "wb") as f:
        #         pickle.dump(tuple1, f)






# import networkx as nx
# import numpy as np
# import pickle as pk
# import os
# # import pandas as pd
# import sys
# # import gurobipy as gp
# # from gurobipy import GRB
# # from new_sampling_methods import *
# # from edge_sampling import *
# # from base_sampling_methods import *
# # from dynamic_sampling import *
# # from pygcn.preprocessing import *
# import math
# import random
# import pickle
#
# sys.setrecursionlimit(1000000)
#
# def sample(ft,adj_sparse,pos_sampled_index1,neg_sampled_index1, pos_index,neg_index):
#     # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
#     sample_len=len(pos_sampled_index1)+len(neg_sampled_index1)
#     adj=np.zeros((sample_len,sample_len))
#     # print(len(graph_sampled_index1))
#     index=[]
#     for i in pos_sampled_index1:
#         index.append(pos_index[i])
#
#     for i in neg_sampled_index1:
#         index.append(neg_index[i])
#
#     for i in range(len(index)):
#         for j in range(i + 1, len(index)):
#             if adj_sparse[index[i]][index[j]] == 1:
#                 adj[i][j]=1
#     ft_list=ft[index]
#     return ft_list,adj
#
# # train_edges=[8,9,5,6,5345]
# # train_lables=[0,0,1,1,1]
# # all = list(range(5))
# #
# # sp_idx = random.sample(all, 5)
# # print(sp_idx)
# #
# # train_edges1=np.array(train_edges)[np.array(sp_idx)]
# # train_labels1=np.array(train_lables)[np.array(sp_idx)]
# # print(train_edges1,train_labels1)
# # #
# # train_lables=[1] * 3 + [0] * 3
# # test_lables = [1] * 4+ [0] * 5
# # print(train_lables)
# # exit()
#
#
# seds=list(range(1,21))
# #METHOD = 'bi-deepwalk2'
# DATASET = 'facebook-data-new-2'
# # DATASET='google+-raw-data-3'
# # DATASET='dblp-data'
# # fnames = os.listdir('E:\\python\\banlance\\code\\facebook-data-new-2\\')
# # fnames = os.listdir('/Users/xiulingwang/Downloads/facebook-data/data/')
# # fnames = os.listdir('E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/')
# # fnames = os.listdir('E:/python/banlance/code/dblp-data/data/')
# # fnames = sorted(fnames)
# # print(fnames)
#
# # G_EGO_USERS = set([])
# # for fname in fnames:
# #     if (fname == '.DS_Store'):
# #         continue
# #     ustring = fname.split('.')[0]
# #     uid = eval(ustring.split('-')[0])
# #     # uid = eval(ustring.split('-')[1])
# #     if uid in G_EGO_USERS:
# #         continue
# #     else:
# #         G_EGO_USERS.add(uid)
# # print(len(G_EGO_USERS))
#
# G_EGO_USERS=['fb']
#
# for sed in seds:
#     # METHOD = 'bi-deepwalk%s' % (sed)
#
#     for ego_user in G_EGO_USERS:
#         # fnames = os.listdir('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/result/')
#         # fnames = sorted(fnames)
#         # n = 'test' + str(ego_user) + '_max.csv'
#         # print(n)
#         # if n in fnames:
#         #     continue
#         # featurename
#         #featname_dir = 'E:/python/banlance/code/facebook-raw-data/' + str(ego_user) + '.featnames'
#         # edges
#         #edges_dir = 'E:/python/banlance/code/facebook-raw-data/' + str(ego_user) + '.edges'
#         # feature
#         # feat_dir = './' + str(ego_user) + '-adj-feat.pkl'
#         # feat_dir = './' + str(ego_user) + '-adj-feat.pkl'
#         # feat_dir = 'E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/' + str(ego_user) + '-adj-feat.pkl'
#         # feat_dir = 'E:/python/banlance/code/dblp-data/data/' + str(ego_user) + '-adj-feat.pkl'
#         # feat_dir = './pokec-adj-feat-processed-45036.pkl'
#         feat_dir = './facebook-adj-feat-procecssed.pkl'
#         f2 = open(feat_dir, 'rb')
#
#         adj, ft = pk.load(f2, encoding='latin1')
#
#         adj0 = np.array(adj,dtype=np.int32)
#
#         g = nx.Graph(adj)
#
#         gender_index=77
#         edu_index = 53
#
#         feat=ft
#
#         # feat=[]
#
#         for i, n in enumerate(g.nodes()):
#             ginfo = ft[n][gender_index]
#             einfo = ft[n][edu_index]
#
#             # print(info)
#             g.nodes[n]['gender'] = ginfo
#             g.nodes[n]['education'] = einfo
#
#
#         np.random.seed(sed)  # make sure train-test split is consistent between notebooks
#         # adj_sparse = nx.to_scipy_sparse_matrix(g)
#         # print(adj_sparse.shape)
#         # print(adj_sparse)
#         num_nodes=g.number_of_nodes()
#
#         pos_index=[]
#         neg_index = []
#         for nd in range(g.number_of_nodes()):
#             if ft[nd][edu_index]==2:
#                 pos_index.append(nd)
#             else:
#                 neg_index.append(nd)
#
#         print(len(pos_index),len(neg_index))
#
#
#         # print(len(pos_index),len(neg_index))
#
#         # exit()
#         sample_cnt=int(1*min(len(pos_index),len(neg_index)))
#         idx_pos = list(range(len(pos_index)))
#         idx_neg = list(range(len(neg_index)))
#         neg_sampled_index1 = random.sample(list(idx_neg), sample_cnt)
#         pos_sampled_index1 = random.sample(list(idx_pos), sample_cnt)
#
#         ft_list1, adj1 = sample(feat, adj0, pos_sampled_index1, neg_sampled_index1, idx_pos, idx_neg)
#
#         g1 = nx.Graph(adj1)
#         # g1.add_edges_from(edge_list1)
#         # g2.add_edges_from(edge_list2)
#
#         adj1 = nx.adjacency_matrix(g1)
#         #
#
#
#         ee = []
#         uu = []
#         eu = []
#         for eg in g1.edges():
#             nd1=eg[0]
#             nd2=eg[1]
#             e1=ft_list1[nd1][edu_index]
#             e2=ft_list1[nd2][edu_index]
#             if e1 + e2 == 2:
#                 # print(e1,e2)
#                 uu.append([eg[0], eg[1]])
#             elif e1 + e2 == 3:
#                 eu.append([eg[0], eg[1]])
#
#             else:
#                 ee.append([eg[0], eg[1]])
#
#
#         print(np.shape(uu),np.shape(eu),np.shape(ee))
#
#         sample_cnt=min((np.shape(uu)[0]+np.shape(ee)[0],np.shape(eu)[0]))
#
#         g_orig = g1
#
#         if (np.shape(uu)[0]+np.shape(ee)[0])>2*np.shape(eu)[0]:
#             exit()
#
#         print(((np.shape(uu)[0]+np.shape(ee)[0])-2*np.shape(eu)[0])/(2*np.shape(eu)[0]))
#
#         num_nodes_ = g1.number_of_nodes()
#
#         idx_edges=[]
#
#         for eg in uu:
#             nd0 = eg[0]
#             nd1 = eg[1]
#
#             # if ft_list1[nd0][edu_index]==1 or ft_list1[nd1][edu_index]==1:
#
#             if (nd0 > nd1):
#                 tmp = nd0
#                 nd0 = nd1
#                 nd1 = tmp
#
#             edge_idx = nd0 * num_nodes_ + nd1
#             idx_edges.append(edge_idx)
#
#         u_nodes = []
#
#         for no in g1.nodes:
#             if ft[no][edu_index] == 1:
#                 u_nodes.append(no)
#
#
#         u_edge_idx = []
#         for i in range(len(u_nodes)):
#             for j in range(i, len(u_nodes)):
#                 u_edge_idx.append(u_nodes[i] * num_nodes_ + u_nodes[j])
#
#         sam_idx=0
#         for ratio in [-0.2,-0.1,0.,0.1,0.2]:
#             sam_idx+=1
#
#             g_orig = g1
#
#             num_add=int((1+ratio)*2*np.shape(eu)[0])
#             if  num_add<0:
#                 print('error')
#                 exit()
#
#
#             print(g1.number_of_edges())
#             print(g_orig.number_of_edges())
#
#
#             # idx_adj = np.array(range(0, num_nodes_ * num_nodes_))
#             # idx_edges = []
#
#             idx_edges = np.array(idx_edges)
#
#             idx_neg_edges = np.setdiff1d(u_edge_idx,idx_edges)
#
#             idx_add_edges = random.sample(list(idx_neg_edges), num_add)
#
#             add_edges = []
#
#             for ed in idx_add_edges:
#                 nd1 = int(ed / num_nodes_)
#                 nd2 = ed % num_nodes_
#                 add_edges.append([nd1, nd2])
#                 add_edges.append([nd2, nd1])
#                 g_orig.add_edges_from(add_edges)
#
#             tuple1=(nx.adjacency_matrix(g_orig),ft_list1)
#
#             with open("../data/fb-interintra/{0}-adj-feat-{1}-{2}-weight.pkl".format(ego_user,str(sam_idx),str(sed)), "wb") as f:
#                 pickle.dump(tuple1, f)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#             #
#         # num_sample=800
#         #
#         # pos_all=list(range(len(pos_index)))
#         # neg_all = list(range(len(neg_index)))
#         # #
#         # pos_len=len(pos_all)
#         # neg_len = len(neg_all)
#         # #
#         # step=int((min(pos_len,neg_len)-800)/100)
#         # print(step,pos_len,neg_len)
#         # # if step>=50:
#         # #     step=50
#         #
#         # for ii in range(-100,100):
#         #     num_sample1=800+ii*step
#         #     num_sample2 = 800-ii *step
#         #     random.seed(sed)
#         #     pos_sampled_index1 = random.sample(pos_all, num_sample1)
#         #     neg_sampled_index1 = random.sample(neg_all, num_sample2)
#         #
#         #     # sampled_index=np.concatenate(( pos_sampled_index1 , neg_sampled_index1))
#         #     # print(len(sampled_index))
#         #
#         #     ft_list1, adj1 = sample(feat, adj0, pos_sampled_index1,neg_sampled_index1, pos_index,neg_index)
#         #     # print(ft_list1)
#         #     # print(np.shape(ft_list1))
#         #
#         #     g1 = nx.Graph(adj1)
#         #     # g1.add_edges_from(edge_list1)
#         #     # g2.add_edges_from(edge_list2)
#         #
#         #     adj1 = nx.adjacency_matrix(g1)
#         #     #
#         #     # adj1_dense = np.array(adj1.todense())
#         #
#         #     tuple1 = (adj1, ft_list1)
#         #
#         #     # if i<0:
#         #     #     label.append(0)
#         #     # else:
#         #     #     label.append(1)
#         #
#         #     with open("../data/fb-edu-gender/{0}-adj-feat-{1}-{2}-edu-gender.pkl".format(ego_user,str(ii),str(sed)), "wb") as f:
#         #         pickle.dump(tuple1, f)
#
