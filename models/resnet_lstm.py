import torch
import torch.nn as nn

from .models import mlp
from .models import rnn


class ResNetLSTM(nn.Module):
    def __init__(self, dim_vid=2048, dim_hidden=512, bidirectional=False, vocab_size=600, dim_word=300, rnn_type="lstm", input_dropout_p=0.2, hidden_dropout_p=0.5, sos_id=2, eos_id=3, freeze_embeds=False):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(ResNetLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_word)
        if freeze_embeds:
            self.embedding.weight.requires_grad = False
        # self.input_dropout = nn.Dropout(input_dropout_p)
        self.vid_encoder = rnn.VidEncoderRNN(
            dim_vid,
            dim_hidden,
            bidirectional=bidirectional,
            input_dropout_p=input_dropout_p,
            rnn_type=rnn_type,
            rnn_dropout_p=hidden_dropout_p)
        self.ques_encoder = rnn.QuesEncoderRNN(
            self.embedding,
            vocab_size,
            dim_hidden,
            dim_word,
            bidirectional=bidirectional,
            input_dropout_p=input_dropout_p,
            rnn_type=rnn_type,
            rnn_dropout_p=hidden_dropout_p)
        # self.decoder = rnn.DecoderRNN(
        #     self.embedding,
        #     vocab_size,
        #     ans_max_len,
        #     dim_hidden * 2,
        #     dim_word,
        #     sos_id,
        #     eos_id,
        #     device,
        #     bidirectional=bidirectional,
        #     input_dropout_p=input_dropout_p,
        #     rnn_type=rnn_type,
        #     rnn_dropout_p=rnn_dropout_p)
        self.mlp = mlp.MLP(dim_hidden * 3, 1, dim_hidden, 2, input_dropout_p=hidden_dropout_p, hidden_dropout_p=hidden_dropout_p)
        self.rnn_type = rnn_type

    def load_embedding(self, pretrained_embedding):
        with torch.no_grad():
            self.embedding.weight.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, vid_feats, question, question_len, answer, answer_len):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        
        # if self.rnn_type == "gru":
        #     vid_encoder_outputs, vid_encoder_hidden = self.vid_encoder(vid_feats)
        #     ques_encoder_outputs, ques_encoder_hidden = self.ques_encoder(question, question_len)
        #     encoder_hidden = torch.cat([vid_encoder_hidden, ques_encoder_hidden], dim=2)
        #     seq_prob, seq_preds = self.decoder(vid_encoder_outputs, encoder_hidden, teacher_forcing, target_variable, opt)
        if self.rnn_type == "lstm":
            vid_encoder_outputs, (vid_encoder_hidden, vid_encoder_cell) = self.vid_encoder(vid_feats)
            ques_encoder_outputs, (ques_encoder_hidden, ques_encoder_cell) = self.ques_encoder(question, question_len)
            ans_encoder_outputs, (ans_encoder_hidden, ans_encoder_cell) = self.ques_encoder(answer, answer_len)
            # encoder_hidden = torch.cat([vid_encoder_hidden, ques_encoder_hidden], dim=2)
            # encoder_cell = torch.cat([vid_encoder_cell, ques_encoder_cell], dim=2)
            encoder_hidden = torch.cat([vid_encoder_hidden, ques_encoder_hidden, ans_encoder_hidden], dim=2)
            # seq_prob, seq_preds = self.decoder(vid_encoder_outputs, (encoder_hidden, encoder_cell), teacher_forcing, target_variable, opt)
            logits = self.mlp(encoder_hidden)
        logits = torch.squeeze(logits)
        probs = torch.sigmoid(logits)
        # return seq_prob, seq_preds
        return logits, probs

    @staticmethod
    def get_fake_inputs(device="cuda:0"):
        bsz = 16
        q = torch.ones(bsz, 32).long().to(device)
        q_l = torch.ones(bsz).fill_(32).long().to(device)
        a = torch.ones(bsz, 32).long().to(device)
        a_l = torch.ones(bsz).fill_(32).long().to(device)
        vid = torch.ones(bsz, 25, 2048).to(device) # imagenet features for each frame
        # vid_l = torch.ones(bsz).fill_(100).long().to(device)
        return vid, q, q_l, a, a_l

if __name__ == '__main__':
    print('torch version:', torch.__version__)
    print('torch cuda version:', torch.version.cuda)
    print('cuda is available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    print("cudnn version:", torch.backends.cudnn.version())

    device = torch.device("cpu")

    model = ResNetLSTM()
    model.to(device)
    test_in = model.get_fake_inputs(device=device)
    test_out = model(*test_in)
    print(test_out.size())