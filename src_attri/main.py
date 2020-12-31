import math
import os
import sys
import time
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy import random
from random import sample
import pandas as pd
from torch.nn.parameter import Parameter
import pickle
from utils import *
from GraphModel import *


def save_walks(network_data):
    all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema, file_name)
    pickle_walks = {'all_walks': all_walks}
    f = open(inter_data_dir+'/'+ allwalks_file_name, 'wb')
    pickle.dump(pickle_walks, f)
    f.close()


def save_vocab():    # generate vocab dict for training
    f = open(inter_data_dir+'/'+allwalks_file_name, 'rb')
    all_walks = pickle.load(f)['all_walks']
    vocab, index2word = generate_vocab(all_walks)
    vocab_train = {'vocab': vocab, 'index2word': index2word}
    f = open(inter_data_dir+'/vocab.pkl', 'wb')
    pickle.dump(vocab_train, f)
    f.close()


def save_vocab_stu_course():
    f = open(inter_data_dir+'/'+allwalks_file_name, 'rb')
    all_walks = pickle.load(f)['all_walks']
    vocab_course, index2word_course, vocab_stu, index2word_stu = generate_vocab_stu_course(all_walks)
    vocab_train_course = {'vocab': vocab_course, 'index2word': index2word_course}
    vocab_train_stu = {'vocab': vocab_stu, 'index2word': index2word_stu}
    f = open(inter_data_dir+'/vocab_course.pkl', 'wb')
    pickle.dump(vocab_train_course, f)
    f = open(inter_data_dir+'/vocab_stu.pkl', 'wb')
    pickle.dump(vocab_train_stu, f)
    f.close()


def save_pairs():
    f = open(inter_data_dir+'/'+allwalks_file_name, 'rb')
    all_walks = pickle.load(f)['all_walks']
    f = open(inter_data_dir+'/vocab_course.pkl', 'rb')
    vocab_course = pickle.load(f)['vocab']
    f = open(inter_data_dir+'/vocab_stu.pkl', 'rb')
    vocab_stu = pickle.load(f)['vocab']
    f = open(inter_data_dir+'/vocab.pkl', 'rb')
    vocab = pickle.load(f)['vocab']
    train_pairs_course, train_pairs_stu = generate_pairs(all_walks, vocab, vocab_course, vocab_stu, args.window_size)
    pairs = {'pairs_course': train_pairs_course, 'pairs_stu': train_pairs_stu}
    f = open(inter_data_dir+'/training_pairs.pkl', 'wb')
    pickle.dump(pairs, f)
    f.close()


def generate_vocab_frequency():
    f = open('../src1/'+inter_data_dir+'/vocab.pkl', 'rb')
    vocab = pickle.load(f)
    index2word = vocab['index2word']
    vocab = vocab['vocab']
    freq = [vocab[index2word[i]].count for i in range(len(index2word))]
    return freq


# generate neighbors for courses and students with their own index
def save_neighbors(network_data):
    f = open(inter_data_dir+'/vocab_course.pkl', 'rb')
    vocab_index = pickle.load(f)
    vocab_course = vocab_index['vocab']
    index2word_course = vocab_index['index2word']
    f = open(inter_data_dir+'/vocab_stu.pkl', 'rb')
    vocab_index = pickle.load(f)
    vocab_stu = vocab_index['vocab']
    index2word_stu = vocab_index['index2word']
    f = open('/research/jenny/RNN/data_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    len_course = len(course_id)

    edge_types = list(network_data.keys())  # [1,2,3,4]
    edge_types.sort()
    num_course = len(index2word_course)
    num_stu = len(index2word_stu)
    edge_type_count = len(edge_types)  # 7
    neighbor_samples = args.neighbor_samples
    neighbors_course = [[[] for __ in range(edge_type_count)] for _ in range(num_course)]  # neighbors for all the courses
    neighbors_stu = [[[] for __ in range(edge_type_count)] for _ in range(num_stu)]   # neighbors for all the students
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        print(edge_types[r])
        for (x, y) in g:
            if int(x) < len_course:  # x is a course, then y is a student
                ix = vocab_course[x].index
                iy = vocab_stu[y].index
                neighbors_course[ix][r].append(iy)
            else:  # x is a student, then y is a course
                ix = vocab_stu[x].index
                iy = vocab_course[y].index
                neighbors_stu[ix][r].append(iy)
        for i in range(num_course):
            if len(neighbors_course[i][r]) == 0:  # no neighbors, num_stu+i, representing the node itself
                neighbors_course[i][r] = [num_stu+i] * neighbor_samples
            elif len(neighbors_course[i][r]) < neighbor_samples:
                neighbors_course[i][r].extend(list(np.random.choice(neighbors_course[i][r], size=neighbor_samples - len(neighbors_course[i][r]))))
            elif len(neighbors_course[i][r]) > neighbor_samples:
                neighbors_course[i][r] = list(sample(neighbors_course[i][r], neighbor_samples))
        for i in range(num_stu):
            if len(neighbors_stu[i][r]) == 0:  # no neighbors, num_course+i, representing the node itself
                neighbors_stu[i][r] = [num_course+i] * neighbor_samples
            if len(neighbors_stu[i][r]) < neighbor_samples:
                neighbors_stu[i][r].extend(list(np.random.choice(neighbors_stu[i][r], size=neighbor_samples - len(neighbors_stu[i][r]))))
            elif len(neighbors_stu[i][r]) > neighbor_samples:
                neighbors_stu[i][r] = list(sample(neighbors_stu[i][r], neighbor_samples))
    f = open(inter_data_dir+'/neighbors.pkl', 'wb')
    neighbors = {'neighbors_course': neighbors_course, 'neighbors_stu': neighbors_stu}
    pickle.dump(neighbors, f)
    f.close()


def training_course_batch_generator(course_loader):
    for i, (batch_x, batch_y) in enumerate(course_loader):
        yield i, batch_x


def training_stu_batch_generator(stu_loader):
    for i, (batch_x, batch_y) in enumerate(stu_loader):
        yield i, batch_x


def train_model(train_bi):

    '''loading training pairs, 1: (course, nodeix, type) and (stu, nodeix, type)'''

    f = open(inter_data_dir+'/vocab_course.pkl', 'rb')
    vocab_index = pickle.load(f)
    vocab_course = vocab_index['vocab']
    index2word_course = vocab_index['index2word']

    f = open(inter_data_dir+'/vocab_stu.pkl', 'rb')
    vocab_index = pickle.load(f)
    vocab_stu = vocab_index['vocab']
    index2word_stu = vocab_index['index2word']

    f = open(inter_data_dir+'/training_pairs.pkl', 'rb')
    pairs = pickle.load(f)
    train_pairs_course = np.array(pairs['pairs_course'], int)  # pairs with node1 as course
    train_pairs_stu = np.array(pairs['pairs_stu'], int)  # pairs with node1 as stu
    f.close()
    vocab_freq = generate_vocab_frequency()  # vocab frequencies for all vocabs

    '''loading neighbors'''

    f = open(inter_data_dir+'/neighbors.pkl', 'rb')
    neighbors = pickle.load(f)
    neighbors_course = np.array(neighbors['neighbors_course'], int)
    neighbors_stu = np.array(neighbors['neighbors_stu'], int)

    '''loading course descriptions'''
    
    course_descriptions = np.load(args.features)

    '''model settings'''

    edge_types = list(train_bi.keys())  # 1-7
    num_course = len(index2word_course)
    num_stu = len(index2word_stu)
    num_nodes = num_course + num_stu
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    dim_course_description = course_descriptions.shape[1]
    dim_hidden = args.hidden_dim
    best_score = None
    patience = 0
    num_iteration = len(train_pairs_course) // batch_size + len(train_pairs_stu) //batch_size

    '''initiate model'''

    print('initiate model')
    model = GATNEModel(num_course, num_stu, embedding_size, embedding_u_size, edge_type_count, dim_a, dim_course_description, dim_hidden)
    nsloss = NSLoss(num_nodes, vocab_freq, num_sampled, embedding_size)

    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-3)
    training_loss_epoch = []
    train_pairs_course_index = torch.IntTensor(np.array(range(train_pairs_course.shape[0])))
    train_pairs_course_index = Data.TensorDataset(train_pairs_course_index, train_pairs_course_index)

    train_pairs_stu_index = torch.IntTensor(np.array(range(train_pairs_stu.shape[0])))
    train_pairs_stu_index = Data.TensorDataset(train_pairs_stu_index, train_pairs_stu_index)

    '''training'''

    for epoch in range(epochs):
        print('-----epoch ' + str(epoch) + '------')
        print('set batches')

        train_pairs_course_loader = Data.DataLoader(dataset=train_pairs_course_index, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
        train_pairs_stu_loader = Data.DataLoader(dataset=train_pairs_stu_index, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

        avg_training_loss = 0.0
        k = 0
        course_batch_generator = training_course_batch_generator(train_pairs_course_loader)
        stu_batch_generator = training_stu_batch_generator(train_pairs_stu_loader)

        batch_shuffle = sample(range(num_iteration), num_iteration)
        for i in batch_shuffle:
            if i < len(train_pairs_course_loader):  # get batch with course as input
                #print('course batch')
                j, batch_x = next(course_batch_generator)
                k += 1
                x = train_pairs_course[:, 0][batch_x]
                y = train_pairs_course[:, 1][batch_x]  # in vocab_all
                t = train_pairs_course[:, 2][batch_x]
                neigh = neighbors_course[list(train_pairs_course[:, 0][batch_x])]
                optimizer.zero_grad()
                embs = model('c', x.tolist(), t.tolist(), neigh, course_descriptions[x.tolist()]).cuda()
                # negative sampling
                loss = nsloss(batch_size, embs, y.tolist()).cuda()
                loss.backward()
                optimizer.step()
                avg_training_loss += loss.item()
                print('--epoch ' + str(epoch) + '--iteration ' + str(k) + '/' + str(num_iteration))
                print('-----course_batch ' + str(j) + '/' + str(len(train_pairs_course_loader)) + '--training_loss ' + str(avg_training_loss / k))
                #break
            else:  # get batch with stu as input
                #print('student batch')
                j, batch_x = next(stu_batch_generator)
                k += 1
                x = train_pairs_stu[:, 0][batch_x]
                y = train_pairs_stu[:, 1][batch_x]  # in vocab_all
                t = train_pairs_stu[:, 2][batch_x]
                neigh = neighbors_stu[list(train_pairs_stu[:, 0][batch_x])]
                optimizer.zero_grad()
                embs = model('s', x.tolist(), t.tolist(), neigh).cuda()
                # negative sampling
                loss = nsloss(batch_size, embs, y.tolist()).cuda()
                loss.backward()
                optimizer.step()
                avg_training_loss += loss.item()
                print('----epoch ' + str(epoch) + '---iteration ' + str(k) + '/' + str(num_iteration))
                print('-----stu_batch ' + str(j) + '/' + str(len(train_pairs_stu_loader)) + '---training_loss ' + str(avg_training_loss/k))
                #break
        avg_training_loss = avg_training_loss / k
        training_loss_epoch.append(avg_training_loss)
        print('----all_training_loss:', training_loss_epoch)


        # record loss
        loss_record = pd.DataFrame(np.array(training_loss_epoch))
        loss_record.columns = ['training_loss']
        loss_record.to_csv(inter_data_dir+'/model/loss_'+str(batch_size)+'.csv', index=False)

        # early stopping
        if best_score:
            if avg_training_loss < best_score:
                best_score = avg_training_loss
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break
        else:
            best_score = avg_training_loss
            #break
            
        '''saving model after each epoch'''
            
        print('saving model')
        torch.save(model, model_name+'_epoch_'+str(epoch)+'.pkl')
        
        '''save course and stu embeddings'''
        
        data_index = torch.IntTensor(np.array(range(len(vocab_course))))
        course_data_index = Data.TensorDataset(data_index, data_index)
        data_loader = Data.DataLoader(dataset=course_data_index, batch_size=batch_size, shuffle=False, num_workers=16,
                                      drop_last=False)
        final_course_embeddings = []
        course_embed = []
        course_des_embed = []
        for (x, y) in data_loader:
            course = x.unsqueeze(1).repeat(1, edge_type_count).view(1, -1).squeeze(0).tolist()
            type = list(range(edge_type_count)) * len(x)
            neigh = neighbors_course[course]
            course_embs = model('c', course, type, neigh, course_descriptions[course]).cuda()
            course_embs = course_embs.view(len(x), edge_type_count, model.embedding_size).cpu().detach().data.numpy()
            final_course_embeddings.extend(course_embs)
            course_embed.extend(F.normalize(model.course_embed[range(0, len(course), edge_type_count, )], dim=1).cpu().detach().data.numpy())
            course_des_embed.extend(F.normalize(model.course_des_embed[range(0, len(course), edge_type_count)], dim=1).cpu().data.numpy())
        final_course_embeddings = np.array(final_course_embeddings)
        course_embed = np.array(course_embed)
        course_des_embed = np.array(course_des_embed)

        data_index = torch.IntTensor(np.array(range(len(vocab_stu))))
        stu_data_index = Data.TensorDataset(data_index, data_index)
        data_loader = Data.DataLoader(dataset=stu_data_index, batch_size=batch_size, shuffle=False, num_workers=16,
                                      drop_last=False)
        final_stu_embeddings = []
        for (x, y) in data_loader:
            stu = x.tolist()
            neigh = neighbors_stu[stu]
            stu_embs = model('s', stu, 1, neigh).cuda()
            stu_embs = stu_embs.view(len(x), model.embedding_size).cpu().detach().data.numpy()
            final_stu_embeddings.extend(stu_embs)
        final_stu_embeddings = np.array(final_stu_embeddings)
        stu_embed = F.normalize(model.student_embeddings, dim=1)
        stu_embed = stu_embed.cpu().data.numpy()
        model_embeddings = {'course_embed': course_embed, 'final_course_embed': final_course_embeddings, 'course_des_embed': course_des_embed,
                            'stu_embed': stu_embed, 'final_stu_embed': final_stu_embeddings}
        f = open(node_embedding_file_name+'_epoch_'+str(epoch)+'.pkl', 'wb')
        pickle.dump(model_embeddings, f)
        f.close()

    '''saving the best model'''
        
    print('saving model')
    torch.save(model, model_name+'.pkl')
    '''save course and stu embeddings'''
    data_index = torch.IntTensor(np.array(range(len(vocab_course))))
    course_data_index = Data.TensorDataset(data_index, data_index)
    data_loader = Data.DataLoader(dataset=course_data_index, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)
    final_course_embeddings = []
    course_embed = []
    course_des_embed = []
    for (x, y) in data_loader:
        course = x.unsqueeze(1).repeat(1, edge_type_count).view(1, -1).squeeze(0).tolist()
        type = list(range(edge_type_count)) * len(x)
        neigh = neighbors_course[course]
        course_embs = model('c', course, type, neigh, course_descriptions[course]).cuda()
        course_embs = course_embs.view(len(x), edge_type_count, model.embedding_size).cpu().detach().data.numpy()
        final_course_embeddings.extend(course_embs)
        course_embed.extend(F.normalize(model.course_embed[range(0, len(course), edge_type_count, )], dim=1).cpu().detach().data.numpy())
        course_des_embed.extend(F.normalize(model.course_des_embed[range(0, len(course), edge_type_count)], dim=1).cpu().data.numpy())
    final_course_embeddings = np.array(final_course_embeddings)
    course_embed = np.array(course_embed)
    course_des_embed = np.array(course_des_embed)

    data_index = torch.IntTensor(np.array(range(len(vocab_stu))))
    stu_data_index = Data.TensorDataset(data_index, data_index)
    data_loader = Data.DataLoader(dataset=stu_data_index, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)
    final_stu_embeddings = []
    for (x, y) in data_loader:
        stu = x.tolist()
        neigh = neighbors_stu[stu]
        stu_embs = model('s', stu, 1, neigh).cuda()
        stu_embs = stu_embs.view(len(x), model.embedding_size).cpu().detach().data.numpy()
        final_stu_embeddings.extend(stu_embs)
    final_stu_embeddings = np.array(final_stu_embeddings)
    stu_embed = F.normalize(model.student_embeddings, dim=1)
    stu_embed = stu_embed.cpu().data.numpy()
    model_embeddings = {'course_embed': course_embed, 'final_course_embed': final_course_embeddings, 'course_des_embed': course_des_embed, 'stu_embed': stu_embed, 'final_stu_embed': final_stu_embeddings}
    f = open(node_embedding_file_name+'.pkl', 'wb')
    pickle.dump(model_embeddings, f)
    f.close()


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    inter_data_dir = args.inter_data_dir
    print(args)
    allwalks_file_name = 'allwalks_numwalks'+str(args.num_walks)+'_walklength'+str(args.walk_length)+'.pkl'

    model_name = inter_data_dir+'/model/GraphModel_' + str(args.batch_size)
    node_embedding_file_name = inter_data_dir+'/model/node_embeddings_' + str(args.batch_size)
    training_data_by_type = load_training_data(file_name)
    print(len(training_data_by_type['1']))
    print(len(training_data_by_type['2']))
    print(len(training_data_by_type['3']))
    print(len(training_data_by_type['4']))
    #save_walks(training_data_by_type)
    #save_vocab()
   # save_vocab_stu_course()
    #save_pairs()
    #save_neighbors(training_data_by_type)

    train_model(training_data_by_type)

