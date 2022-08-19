import operator
from queue import PriorityQueue
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import Attention


class VidEncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_type='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_type (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(VidEncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU

        self.rnn = self.rnn(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden

class QuesEncoderRNN(nn.Module):
    def __init__(self, embedding_layer, vocab_size, dim_hidden, dim_word, input_dropout_p, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_type='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_type (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(QuesEncoderRNN, self).__init__()
        self.embedding = embedding_layer
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU

        self.rnn = self.rnn(dim_word, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        # nn.init.xavier_normal_(self.vid2hid.weight)
        # TODO: init embedding layer with Glove
        pass

    def forward(self, question, question_len):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        # batch_size, seq_len, dim_vid = vid_feats.size()
        # vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        ques_embeds = self.embedding(question)
        ques_embeds = self.input_dropout(ques_embeds)
        ques_embeds = nn.utils.rnn.pack_padded_sequence(ques_embeds, question_len, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(ques_embeds)
        return output, hidden


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                embedding_layer,
                vocab_size,
                max_len,
                dim_hidden,
                dim_word,
                sos_id,
                eos_id,
                device,
                input_dropout_p,
                n_layers=1,
                rnn_type='gru',
                bidirectional=False,
                rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding_layer
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.bidirectional_encoder = bidirectional
        self.max_length = max_len 
        dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU
        self.rnn = self.rnn(
            dim_word,
            dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.hid2emd = nn.Linear(dim_hidden, dim_word)
        self.out = nn.Linear(dim_word, vocab_size)
        self.vocab_size = vocab_size
        self._init_weights()


    def _init_weights(self):
        """ init the weight of some layers
        """
        # nn.init.xavier_normal_(self.out.weight)
        self.out.weight = self.embedding.weight # Tie the parameters of the output layer and the embedding layer

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                teacher_forcing,
                target_variable=None,
                opt={}):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        self.rnn.flatten_parameters()
        if teacher_forcing:
            # use targets as rnn inputs
            assert target_variable is not None

            seq_logprobs = []
            seq_preds = None
            targets_emb = self.embedding(target_variable)
            targets_emb = self.input_dropout(targets_emb)
            for i in range(self.max_length - 1):
                decoder_input = targets_emb[:, i, :].unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(self.hid2emd(decoder_output.squeeze(1))), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            seq_logprobs = torch.cat(seq_logprobs, 1)

        elif opt["decoding"] == "greedy":
            seq_logprobs, seq_preds = self.greedy_decode(batch_size, decoder_hidden)
        elif opt["decoding"] == "beam":
            seq_logprobs, seq_preds = self.beam_decode(batch_size, decoder_hidden, opt=opt)
        return seq_logprobs, seq_preds

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def beam_decode(self, batch_size, decoder_hiddens, encoder_outputs=None, opt={}):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        beam_width = opt["beam_size"]
        topk = 1  # how many sentences do you want to generate
        seq_preds = []

        # decoding goes sentence by sentence
        for idx in range(batch_size):
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
            
            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[self.sos_id]]).to(self.device)

            # Number of sentences to generate
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

                if n.wordid.item() == self.eos_id and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_input = self.embedding(decoder_input)
                decoder_input = self.input_dropout(decoder_input)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(self.hid2emd(decoder_output.squeeze(1))), dim=1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(logprobs, beam_width)
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

            seq_preds.append(utterances)

        return None, seq_preds

    def greedy_decode(self, batch_size, decoder_hidden):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        seq_logprobs = []
        seq_preds = []
        decoder_input = torch.LongTensor([[self.sos_id] for _ in range(batch_size)]).to(self.device)

        for t in range(self.max_length):
            decoder_input = self.embedding(decoder_input)
            decoder_input = self.input_dropout(decoder_input)
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
            seq_logprobs.append(logprobs.unsqueeze(1))
            _, topi = logprobs.topk(1)  # get candidates
            seq_preds.append(topi)

            decoder_input = topi.detach()

        seq_logprobs = torch.cat(seq_logprobs, dim=1)
        seq_preds = torch.cat(seq_preds, dim=1)
        return seq_logprobs, seq_preds


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return bool(random.getrandbits(1))


class DecoderRNNWithAtt(DecoderRNN):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 sos_id=1,
                 eos_id=0,
                 n_layers=1,
                 rnn_type='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1):
        super(DecoderRNNWithAtt, self).__init__()

        self.bidirectional_encoder = bidirectional

        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, dim_word)
        self.attention = Attention(self.dim_hidden)
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU
        self.rnn = self.rnn(
            self.dim_hidden + dim_word,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                opt={}):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []
        self.rnn.flatten_parameters()
        if mode == 'train':
            # use targets as rnn inputs
            targets_emb = self.embedding(targets)
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            seq_logprobs = torch.cat(seq_logprobs, 1)

        elif mode == 'inference':
            if beam_size > 1:
                return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length - 1):
                context = self.attention(
                    decoder_hidden.squeeze(0), encoder_outputs)

                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).cuda()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()

                else:
                    # sample according to distribuition
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, it)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()

                seq_preds.append(it.view(-1, 1))

                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)

        return seq_logprobs, seq_preds