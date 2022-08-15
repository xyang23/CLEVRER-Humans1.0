#!/usr/bin/env python
# coding: utf-8
from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import random
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from queue import PriorityQueue
import operator

# SPECIFY SINGLE OBJECT OR NOT HERE
is_single = True
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1

# ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.n_len_trajectory = 0 # Not sure. 
        
    def addTrajectory(self, sentence):
        
        self.n_len_trajectory = len(sentence) # remove + 1 here because we do not need to add SOS or EOS
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs_new(lang1, lang2, reverse=False, is_single=True):
    print("Reading lines...")
    if is_single:
        pairs = pickle.load(open("/viscam/u/xyang23/lstm/pair_data_w_property_10-19.pkl", 'rb')) # single obj
    else:
        pairs = pickle.load(open('/viscam/u/xyang23/lstm/checkpoints/two_obj_data.p', 'rb')) # double obj
    
    # Read the file and split into lines
    for i in range(len(pairs)):
        pairs[i][1] =  normalizeString(pairs[i][1])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1) 
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 50

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False, is_single=True):
 
    input_lang, output_lang, pairs = readLangs_new(lang1, lang2, reverse, is_single) # change to input_lang.name be trajectory, output_lang.name be sentence
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        #input_lang.addSentence(pair[0]) # commented this line because we are not using one-hot matrix for trajectory
        input_lang.addTrajectory(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_len_trajectory)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs_all = prepareData('trajectory', 'eng', False, is_single)
training_split = (len(pairs_all)// 10) * 9
pairs = pairs_all

print(len(pairs_all), len(pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size) #debug- apply a neural network, 64 dimension, input_size x hidden_size 
        self.encoding = nn.Linear(input_size, hidden_size) 
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_size, hidden_size) # GRU set batch_first

    def forward(self, inputs, hidden):
        #embedded = self.embedding(inputs).view(1, 1, -1)
        #output = embedded
        #output, hidden = self.gru(output, hidden)
       
        inputs = inputs.view(1, 1, -1)  
        encoded = self.encoding(inputs)
        output = self.activation(encoded)
    #    print(output.size(), hidden.size())

        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.max_length_input = 128
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length_input)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
    
        attn_weights = F.softmax(
             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#        attn_weights = torch.ones(1, 1).to(device)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[14]:


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromTrajectory(lang, sentence):
    indexes = sentence # As the original sentence is the trajectory
    #indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.float, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromTrajectory(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    max_length_input = 128
    encoder_outputs = torch.zeros(max_length_input, encoder.hidden_size, device=device) # 10 x 256, 3072 x 256

    loss = 0
    for ei in range(min(input_length, max_length_input)):
        encoder_output, encoder_hidden = encoder(
            input_tensor.view(128, -1)[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, save_every=1000, eval_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    checkpoint_model = []
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)] # If there are 100 iteration, there are 100 training pairs
    criterion = nn.NLLLoss()
    
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % save_every == 0:
            checkpoint_model.append([encoder, decoder, loss])
            with open('/viscam/u/xyang23/lstm/checkpoints/double_model_22_3_1-5.p', 'wb') as f:
                pickle.dump(checkpoint_model, f)
                print("model saved")


        if iter % eval_every == 0:
            evaluateRandomly(encoder, decoder, n=15)
            
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    def __lt__(self, item1):
        return item1


def beam_decode(decoder, target_tensor, decoder_hiddens, encoder_outputs=None, beam_width=10, topk=3):
    decoded_batch = []
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs

        # Start with the start of the sentence token
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_input = decoder_input

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
          
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


def evaluate_topk(encoder, decoder, sentence, max_length=MAX_LENGTH, topk=10):
    with torch.no_grad():
        input_tensor = tensorFromTrajectory(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        max_length_input = 128
        encoder_outputs = torch.zeros(max_length_input, encoder.hidden_size, device=device)

        #encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        #encoder_outputs = encoder_outputs.view(1, -1)
        for ei in range(min(input_length, max_length_input)):
            encoder_output, encoder_hidden = encoder(
                input_tensor.view(128, -1)[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words_list = []
        
        sentences = beam_decode(decoder, decoder_input, decoder_hidden, encoder_outputs, beam_width=20, topk=topk)[0]
        for sentence in sentences:
            decoded_words = []
            for word in sentence:
                word = word.item()
                if word == EOS_token:
                    decoded_words.append('<EOS>')
                    decoded_words_list.append(decoded_words)
                    break
                elif word == SOS_token:
                    continue
                else:
                    decoded_words.append(output_lang.index2word[word])

        return decoded_words_list

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromTrajectory(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        encoder_outputs = encoder_outputs.view(1, -1)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=100):
    for i in range(n):
        pair = random.choice(pairs)
#         print('>', pair[0])
        print('=', pair[1])
        output_sentences = evaluate_topk(encoder, decoder, pair[0])
        for output_words in output_sentences:
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        print('\n')


# Correctness Check



all_colors = ['gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow', 'gold', 'silver']
all_shapes = ['cube', 'sphere', 'cylinder', 'ball']
all_materials = ['metal', 'rubber', 'magenta', 'metallic']
map_color = {'gray' : 0, 'red' : 1, 'blue' : 2, 'green': 3, 'brown': 4, 'cyan' : 5, 'purple' : 6, 'yellow' : 7, 'gold': 7, 'silver': 0};
map_shape = {'cube': 0, 'sphere': 1, 'cylinder': 2, 'ball': 1}
map_material = {'metal': 0, 'rubber': 1, 'magenta': 0, 'metallic': 0}
    
def grammar_check(output_words, is_single=True):
    # check if the output sentence is gramatically correct
    end_word_error_list = {"from", "to", "is", "was", "still", "had", "the", "at", "not", "a", "been"}
    if output_words[-2] in end_word_error_list:    
        # ignore the false output words
        return False
    for word_idx, word in enumerate(output_words): # eliminate continued word
        if word_idx > 0 and output_words[word_idx - 1] == word:
            return False
    return True

def hash_compare_obj_dict(obj1, obj2):
    # take in two dict for obj comparison
    if map_color[obj1['color']] != map_color[obj2['color']] or map_shape[obj1['shape']] != map_shape[obj2['shape']]:
        return False
    if 'material' in obj1.keys() and 'material' in obj2.keys() and map_material[obj1['material']] != map_material[obj2['material']]:
        return False
    return True
    
def factual_check(output_words, input_object, is_single=True):
    # check if the output sentence is factually correct
    # we first parse the sentence
    # adopted from data_prep
    colors = []
    shapes = []
    materials = []
    colors_idx = []
    shapes_idx = []
    materials_idx = []
    for word_idx, word in enumerate(output_words):
        if word in all_colors:
            colors.append(word)
            colors_idx.append(word_idx)
        elif word in all_shapes:
            shapes.append(word)
            shapes_idx.append(word_idx)
        elif word in all_materials:
            materials.append(word)
            materials_idx.append(word_idx)
    if is_single:
        # for single sentences
        if not isinstance(input_object, dict):
            raise Exception("input_object should be a dictionary for single sentences")
        if len(colors) != 1 or len(shapes) != 1 or len(materials) > 1:
            # first check the number of color/shape/material is correct
            return False
        if colors_idx[0] > shapes_idx[0]:
            # check order
            return False
        if len(materials) == 1 and materials_idx[0] > shapes_idx[0]:
            return False
        obj1 = {'color': colors[0], 'shape': shapes[0]}
        if len(materials) == 1:
            if abs(materials_idx[0] - colors_idx[0]) != 1 or shapes_idx[0] - max(materials_idx[0], colors_idx[0]) != 1:
                # if not continuous
                return False
            obj1['material'] = materials[0]
        elif shapes_idx[0] - colors_idx[0] != 1:
            return False
        return hash_compare_obj_dict(obj1, input_object)         
    else:
        # for double sentences
        if not isinstance(input_object, list):
            raise Exception("input_object should be a list for two dictionary for double sentences")
        if len(colors) != 2 or len(shapes) != 2 or len(materials) > 2:
            return False  
        if colors_idx[0] > shapes_idx[0] or colors_idx[1] > shapes_idx[1]:
            return False
        if len(materials)  == 2 and (materials_idx[0] > shapes_idx[0] or materials_idx[1] > shapes_idx[1]):
            return False
        obj1 = {'color': colors[0], 'shape': shapes[0]}
        obj2 = {'color': colors[1], 'shape': shapes[1]}
        if len(materials) == 0:
            if shapes_idx[0] - colors_idx[0] != 1:
                return False
            if shapes_idx[1] - colors_idx[1] != 1:
                return False  
            if colors_idx[1] - shapes_idx[0] == 1:
                # exclude "red cube blue cube"
                return False
        elif len(materials) == 1:
            # locate the material for the correct object
            if materials_idx[0] < shapes_idx[0]:
                obj1['material'] = materials[0]
                if abs(materials_idx[0] - colors_idx[0]) != 1 or shapes_idx[0] - max(materials_idx[0], colors_idx[0]) != 1:
                # if not continuous
                    return False
                if shapes_idx[1] - colors_idx[1] != 1:
                    return False
                if colors_idx[1] - shapes_idx[0] == 1:
                # exclude "red cube blue cube"
                    return False
            else:
                obj2['material'] = materials[0]
                if abs(materials_idx[0] - colors_idx[1]) != 1 or shapes_idx[1] - max(materials_idx[0], colors_idx[1]) != 1:
                    return False
                if shapes_idx[0] - colors_idx[0] != 1:
                    return False
                if min(materials_idx[0], colors_idx[1]) - shapes_idx[0] == 1:
                # exclude "red cube blue cube"
                    return False
        elif len(materials) == 2:
            obj1['material'] = materials[0]
            obj2['material'] = materials[1]
            if abs(materials_idx[0] - colors_idx[0]) != 1 or shapes_idx[0] - max(materials_idx[0], colors_idx[0]) != 1:
                return False
            if abs(materials_idx[1] - colors_idx[1]) != 1 or shapes_idx[1] - max(materials_idx[1], colors_idx[1]) != 1:
                return False
            if min(materials_idx[1], colors_idx[1]) - shapes_idx[0] == 1:
                # exclude "red cube blue cube"
                return False
        if (hash_compare_obj_dict(obj1, input_object[0]) and hash_compare_obj_dict(obj2, input_object[1])):
            return True
        if (hash_compare_obj_dict(obj2, input_object[0]) and hash_compare_obj_dict(obj1, input_object[1])):
            return True
        return False


input_list = [{'object_id': 1, 'color': 'green', 'material': 'metal', 'shape': 'sphere'}, {'object_id': 2, 'color': 'yellow', 'material': 'metal', 'shape': 'cylinder'}]
sentence = "the green ball stood motionless in the path of the yellow cylinder"
print(factual_check(sentence.split(' '), input_list, is_single=False))
print(grammar_check(sentence.split(' '), is_single=False))

hidden_size = 128
encoder1 = EncoderRNN(input_lang.n_len_trajectory//128, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.3).to(device)
print("initialized.")
trainIters(encoder1, attn_decoder1, 20000, print_every=1000, save_every=1000, eval_every=1000, learning_rate=0.01)
