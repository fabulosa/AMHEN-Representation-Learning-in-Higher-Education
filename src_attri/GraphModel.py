import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy import random
from torch.autograd import Variable
import pickle
from utils import *


class GATNEModel(nn.Module):
    def __init__(self, num_courses, num_students, embedding_size, embedding_u_size, edge_type_count, dim_a, dim_course_descrip, hidden_size):
        super(GATNEModel, self).__init__()
        self.num_courses = num_courses
        self.num_students = num_students
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a
        self.course_embed = None
        self.course_des_embed = None

        self.course_h1 = nn.Linear(dim_course_descrip, hidden_size).cuda()
        self.course_h_relu = nn.ReLU().cuda()
        self.course_h2 = nn.Linear(hidden_size, embedding_size).cuda()
        self.course_trans = nn.Parameter(torch.FloatTensor(dim_course_descrip, embedding_size).cuda(), requires_grad=True)

        # base embeddings for courses, if not include course features
        #self.course_embeddings = nn.Parameter(torch.FloatTensor(num_courses, embedding_size).cuda(), requires_grad=True)  # (*, d)
        # base embeddings for students
        self.student_embeddings = nn.Parameter(torch.FloatTensor(num_students, embedding_size).cuda(), requires_grad=True) # (*, d)
        # course has grade type embeddings
        self.course_type_embeddings = nn.Parameter(torch.FloatTensor(num_courses, edge_type_count, embedding_u_size).cuda(), requires_grad=True) # (*, m, s)
        # student has a type for embeddings
        self.student_type_embeddings = nn.Parameter(torch.FloatTensor(num_students, 1, embedding_u_size).cuda(), requires_grad=True)

        # courses
        self.trans_weights_c = nn.Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size).cuda(), requires_grad=True)  # (m, s, d)
        # students
        self.trans_weights_s = nn.Parameter(torch.FloatTensor(1, embedding_u_size, embedding_size).cuda(), requires_grad=True)  # (1, s, d)

        self.trans_weights_s1 = nn.Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, dim_a).cuda(), requires_grad=True)  # (m, s, da)
        self.trans_weights_s2 = nn.Parameter(torch.FloatTensor(edge_type_count, dim_a, 1).cuda(), requires_grad=True)  # (m, da, 1)

        self.reset_parameters()

    def reset_parameters(self):
        #self.course_embeddings.data.uniform_(-1.0, 1.0)
        self.course_trans.data.uniform_(-1.0, 1.0)
        self.student_embeddings.data.uniform_(-1.0, 1.0)
        self.course_type_embeddings.data.uniform_(-1.0, 1.0)
        self.student_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights_c.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input_type, train_inputs, train_types, node_neigh, course_descrip=None):

        if input_type == 'c':  # course as input, self-attention

            course_descrip = Variable(torch.FloatTensor(course_descrip), requires_grad=False).cuda()

            #course_embed = self.course_embeddings[train_inputs]
            self.course_embed = self.course_h1(course_descrip)
            self.course_embed = self.course_h_relu(self.course_embed)
            self.course_embed = self.course_h2(self.course_embed)

            self.course_des_embed = torch.matmul(course_descrip, self.course_trans)

            args = parse_args()
            course_embed_neighbors = Variable(torch.FloatTensor(len(train_inputs), self.edge_type_count, args.neighbor_samples, self.embedding_u_size), requires_grad=True).cuda()
            for i in range(self.edge_type_count):
                neigh_i = node_neigh[:, i].reshape(1, len(train_inputs) * args.neighbor_samples).squeeze(0).tolist()
                temp_stu = self.student_type_embeddings[:, 0]
                temp_course = self.course_type_embeddings[:, i]
                temp = torch.cat((temp_stu, temp_course), 0)
                neigh_i_emb = temp[neigh_i].contiguous().view(len(train_inputs), args.neighbor_samples, self.embedding_u_size)
                course_embed_neighbors[:, i] = neigh_i_emb
            course_type_embed = torch.mean(course_embed_neighbors, dim=2)

            trans_w = self.trans_weights_c[train_types]
            trans_w_s1 = self.trans_weights_s1[train_types]
            trans_w_s2 = self.trans_weights_s2[train_types]

            attention = F.softmax(torch.matmul(torch.tanh(torch.matmul(course_type_embed, trans_w_s1)), trans_w_s2).squeeze(2), dim=1).unsqueeze(1)
            course_type_embed = torch.matmul(attention, course_type_embed)
            course_embed = self.course_embed + torch.matmul(course_type_embed, trans_w).squeeze(1) + self.course_des_embed

            last_course_embed = F.normalize(course_embed, dim=1)
            return last_course_embed

        elif input_type == 's':  # student as input, skip-gram
            student_embed = self.student_embeddings[train_inputs]
            args = parse_args()
            student_embed_neighbors = Variable(torch.FloatTensor(len(train_inputs), self.edge_type_count, args.neighbor_samples, self.embedding_u_size), requires_grad=True).cuda()
            for i in range(self.edge_type_count):
                neigh_i = node_neigh[:, i].reshape(1, len(train_inputs) * args.neighbor_samples).squeeze(0).tolist()
                temp_course = self.course_type_embeddings[:, i]
                temp_stu = self.student_type_embeddings[:, 0]
                temp = torch.cat((temp_course, temp_stu), 0)
                neigh_i_emb = temp[neigh_i].contiguous().view(len(train_inputs), args.neighbor_samples, self.embedding_u_size)
                student_embed_neighbors[:, i] = neigh_i_emb
            student_type_embed = torch.mean(torch.mean(student_embed_neighbors, dim=2), dim=1)
            trans_w = self.trans_weights_s[0]
            student_embed = student_embed + torch.matmul(student_type_embed, trans_w).squeeze(1)
            last_student_embed = F.normalize(student_embed, dim=1)
            return last_student_embed


class NSLoss(nn.Module):
    def __init__(self, num_nodes, node_freq, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes  # courses + students
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = nn.Parameter(torch.FloatTensor(num_nodes, embedding_size).cuda())  # context weights (*, d)
        # node freq is a list of node frequency for all nodes
        self.node_freq = np.array(node_freq) ** (3/4)
        total_freq = np.sum(node_freq)
        self.sample_weights = torch.Tensor([self.node_freq[k] / total_freq for k in range(num_nodes)])

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, n, embs, label):
        #n = len(input)
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1)))
        negs = torch.multinomial(self.sample_weights, self.num_sampled * n, replacement=True).cuda()
        noise = torch.neg(self.weights[negs].view(n, self.num_sampled, -1))
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


