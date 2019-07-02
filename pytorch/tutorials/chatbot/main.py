from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

#printLines(os.path.join(corpus, "movie_lines.txt"))

# Read each line as dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj # indexed by 'lineID' field
    return lines

# Group lines by conversation based on 'movie_conversations.txt'
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # from string representation of python list of lineIds
            lineIds = eval(convObj["utteranceIDs"])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# Sentence pairs from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # filter empty samples <- not sure why this happens
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

datafile = os.path.join(corpus, "formatted_movie_lines.txt")

generateFormattedFile = False
if generateFormattedFile:
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["characterID", "character2ID", "movieID", "utteranceIDs"]


    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)

    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # print sample of lines
    print("\nSample lines from file:")
    printLines(datafile)


# Voc: class for word - index mapping
PAD_token = 0 # pad short sentences
SOS_token = 1 # start of sentence
EOS_token = 2 # end of sentence

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # PAD, SOS, EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True # trim once

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {}/{} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words)/len(self.word2index)))

        # give new indices for keep words
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)


MAX_LENGTH = 10

# unicode -> ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# lowercase, trim, remove non-letters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# read sentence pairs and return voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# returns True iff both sentences < MAX_LENGTH
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

'''
print("\npairs:")
for pair in pairs[:10]:
    print(pair)
'''

MIN_COUNT = 3 # minimum word count threshold
def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}".format(len(pairs), len(keep_pairs)))
    return keep_pairs

pairs = trimRareWords(voc, pairs, MIN_COUNT)

def indexesFromSentence(voc, sentence):
    # list of indices
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    # transpose implicitly
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):

    # (B(batch size), L(seq len)) <- not uniform length
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    # (B)
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # (L, B)
    padList = zeroPadding(indexes_batch)

    # (L, B)
    padVar = torch.LongTensor(padList)

    return padVar, lengths

def outputVar(l, voc):

    # (B, L) <- not uniform length
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    # ()
    max_target_len = max([len(indexes) for indexes in indexes_batch])

    # (L, B)
    padList = zeroPadding(indexes_batch)

    # (L, B)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    
    # (L, B)
    padVar = torch.LongTensor(padList)

    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):

    # list of B pairs
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)

    # <dimensions>
    # inp: (L, B)
    # lengths: (B)
    # output: (L, B)
    # mask: (L, B)
    # max_target_len: ()
    return inp, lengths, output, mask, max_target_len

'''
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len", max_target_len)
'''

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # input dim is also hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                            dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):

        # (L, B)->(L, B, H)
        embedded = self.embedding(input_seq)
        
        # no calculation on zero paddings
        packed = nn.utils.rnn.pack_padded_sequences(embedded, input_lengths)

        # outputs: (L, B, 2H)
        # hidden: (n_layers * 2, B, H)
        outputs, hidden = self.gru(packed, hidden) # outputs and final hidden state
        outputs, _ = nn.utils.rnn.pad_packed_sequences(outputs)

        # sum outputs for 2 directions
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs: (L, B, H)
        # hidden: (n_layers * 2, B, H)
        return outputs, hidden

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        def dot_score(self, hidden, encoder_output):
            return torch.sum(hidden * encoder_output, dim=2)

        def general_score(self, hidden, encoder_output):
            energy = self.attn(encoder_output)
            return torch.sum(hidden * energy, dim=2)

        def concat_score(self, hidden, encoder_output):
            energy = self.attn(
                torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
            ).tanh()
            return torch.sum(self.v * energy, dim=2)

        def forward(self, hidden, encoder_outputs):

            # (L, B)
            if self.method == 'general':
                attn_energies = self.general_score(hidden, encoder_outputs)
            elif self.method == 'concat':
                attn_energies = self.concat_score(hidden, encoder_outputs)
            elif self.method == 'dot':
                attn_energies = self.dot_score(hidden, encoder_outputs)

            # (B, L)
            attn_energies = attn_energies.t()

            # softmax over L, unsqueeze -> (B, 1, L)
            return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.W_concat = nn.Linear(hidden_size * 2, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):

        # input_step: (1, B)
        # last_hidden: (n_layers, B, H)
        # encoder_outputs: (L, B, H)

        # (1, B, H)
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # rnn_output: (1, B, H)
        # hidden: (n_layers, B, H)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # (B, 1, L)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        # (B, 1, L) batchMatMul (B, L, H) -> (B, 1, H)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # (B, H)
        rnn_output = rnn_output.squeeze(0)

        # (B, H)
        context = context.squeeze(1)

        # (B, 2H)
        concat_input = torch.cat((rnn_output, context), 1)

        # (B, H)
        concat_output = torch.tanh(self.W_concat(concat_input))

        # (B, V) V: output vocabulary size
        output = self.W_out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden



# Training procedure

def maskNLLLoss(inp, target, mask):
    
    # <dimensions>
    # inp (decoder output): (B, V)
    # target: (1, B)
    # mask: (1, B)
    
    # nTotal: ()
    nTotal = mask.sum()

    # target^T: (B, 1)
    # gather_result[b][0]: inp[b][target^T[b][0]]: (B, 1)
    # crossEntropy: (B)
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    # indexing (with gather()) is equivalent to dot product with one-hot vector

    # loss: ()
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)

    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # encoder forward pass
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # create initial decoder input
    decoder_input = toarch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # set initial decoder hidden state to encoder final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # randomly determine whether to use teacher forcing for this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # forward pass trough time, calculate loss
    if use_teacher_forcing:
        for t in range(max_target_len):
            # decoder forward pass
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # teacher forcing

            # (1, B)
            decoder_input = target_variable[t].view(1, -1)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            # decoder forward pass
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )            
            # feed previous output

            # topi: most likely word index (use last dimension: V)
            # -> shape (B, 1)
            _, topi = decoder_output.topk(1)

            # (1, B)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Backprop
    loss.backward()

    # Clip gradients (in place)
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


