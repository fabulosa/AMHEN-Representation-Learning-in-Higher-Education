import argparse
from collections import defaultdict
import pickle
import networkx as nx
import numpy as np
from gensim.models.keyedvectors import Vocab
from six import iteritems
#from sklearn.metrics import (auc, f1_score, precision_recall_curve, roc_auc_score)
from walk import RWGraph


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='train.txt',
                        help='Training set path')

    parser.add_argument('--inter_data_dir', type=str, default='numwalks10_walklength5',
                        help='Intermediate data directory')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--walk-length', type=int, default=5,
                        help='Length of walk per source. Default is 5.')

    parser.add_argument('--features', type=str, default='course_description.npy',
                        help='Input node features')

    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epoch. Default is 10.')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--schema', type=str, default='s-c-s, c-s-c',
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=50,
                        help='Number of edge embedding dimensions. Default is 50.')

    parser.add_argument('--att-dim', type=int, default=100,
                        help='Number of attention dimensions. Default is 100.')

    parser.add_argument('--hidden-dim', type=int, default=500,
                        help='Number of attention dimensions. Default is 500.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--negative-samples', type=int, default=10,
                        help='Negative samples for optimization. Default is 10.')

    parser.add_argument('--neighbor-samples', type=int, default=20,
                        help='Neighbor samples for aggregation. Default is 20.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')

    return parser.parse_args()


# loading training data by edge type {edge_type: [(node1, node2),...], ...}
def load_training_data(f_name):
    print('loading data from:', f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if int(words[0]) <= 2:
                words[0] = '1'
            elif int(words[0]) <= 5: 
                words[0] = '2'
            elif int(words[0]) == 6:
                words[0] = '3'
            else:
                words[0] = '4'
            if words[0] not in edge_data_by_type:# A or B
                edge_data_by_type[words[0]] = list()# C or D or F
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total number of training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total number of validation nodes: ' + str(len(all_nodes)))
    return true_edge_data_by_type, false_edge_data_by_type


def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


# generate a graph for each edge type: [(node1, node2), ...] -> Graph (weight)
def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('_')[0]
        y = edge_key.split('_')[1]
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G


# generate random walks for all edge types
def generate_walks(network_data, num_walks, walk_length, schema, file_name):
    if schema is not None:
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []   # each row is simulated works in a layer
    layers = list(network_data.keys())
    layers.sort()
    for layer_id in layers:
        print('the ' + str(layer_id) + ' layer')
        tmp_data = network_data[layer_id]

        # for each edge type, construct a RWGraph, generate random works for training sequences
        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks


def generate_pairs(all_walks, vocab, vocab_course, vocab_stu, window_size):  # pairs (course, index2, layer_id) and (stu, index2, layer_id)
    f = open('/research/jenny/RNN/data_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    len_course = len(course_id)
    pairs_course = []  # (course, index2, layer_id)
    pairs_stu = []   # (stu, index2, layer_id)
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        print('layer '+str(layer_id))
        for walk in walks:
            for i in range(len(walk)):
                if int(walk[i]) < len_course:  # the first node is a course
                    for j in range(1, skip_window + 1):
                        if i - j >= 0:
                            pairs_course.append((vocab_course[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                        if i + j < len(walk):
                            pairs_course.append((vocab_course[walk[i]].index, vocab[walk[i + j]].index, layer_id))
                else:  # the first node is a student
                    for j in range(1, skip_window + 1):
                        if i - j >= 0:
                            pairs_stu.append((vocab_stu[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                        if i + j < len(walk):
                            pairs_stu.append((vocab_stu[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs_course, pairs_stu


# generate vocab for all nodes, for context embeddings of skip-gram
def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)
    print("count")
    for i, walks in enumerate(all_walks):
        print('layer '+str(i))
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1
    print('generate vocab')
    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


# generate vocab from all_walks for students and courses, respectively, for the input skip-gram
def generate_vocab_stu_course(all_walks):
    f = open('/research/jenny/RNN/data_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    len_course = len(course_id)

    index2word_course = []
    raw_vocab_course = defaultdict(int)
    index2word_stu = []
    raw_vocab_stu = defaultdict(int)
    for i, walks in enumerate(all_walks):
        print('layer ' + str(i))
        for walk in walks:
            for word in walk:
                word_int = int(word)
                if word_int < len_course:  # course
                    raw_vocab_course[word] += 1
                else:
                    raw_vocab_stu[word] += 1
    print('generate vocab')
    vocab_course = {}
    for word, v in iteritems(raw_vocab_course):
        vocab_course[word] = Vocab(count=v, index=len(index2word_course))
        index2word_course.append(word)
    index2word_course.sort(key=lambda word: vocab_course[word].count, reverse=True)
    for i, word in enumerate(index2word_course):
        vocab_course[word].index = i

    vocab_stu = {}
    for word, v in iteritems(raw_vocab_stu):
        vocab_stu[word] = Vocab(count=v, index=len(index2word_stu))
        index2word_stu.append(word)
    index2word_stu.sort(key=lambda word: vocab_stu[word].count, reverse=True)
    for i, word in enumerate(index2word_stu):
        vocab_stu[word].index = i

    return vocab_course, index2word_course, vocab_stu, index2word_stu
