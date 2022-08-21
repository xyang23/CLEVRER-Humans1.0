from distutils.util import strtobool
import inspect
import json
import os
import pickle

import torch
import torch.nn as nn


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, batched_seqs, offset=4):
    out = []
    for seq in batched_seqs:
        tokens = []
        for ix in seq[0]: # beam search decoding generates the top 1 answer for now
            ix = ix.item()
            if ix >= offset: # Don't include <sos>, <eos>, <unk> and <pad>
                tokens.append(ix_to_word[ix])
            elif ix == 0:
                break
        txt = " ".join(tokens)
        out.append(txt)
    return out


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                         mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def pprint_json(obj):
    print(json.dumps(obj, indent=2, sort_keys=True))


def save_pickle(data, data_path):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def files_exist(filepath_list):
    """check whether all the files exist"""
    for ele in filepath_list:
        if not os.path.exists(ele):
            return False
    return True

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def symlink(source, link):
    try:
        os.symlink(source, link)
    except FileExistsError:
        os.unlink(link)
        os.symlink(source, link)


def print_version_info():
    print('torch version:', torch.__version__)
    print('torch cuda version:', torch.version.cuda)
    print('cuda is available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    print("cudnn version:", torch.backends.cudnn.version())


def str2bool(v):
    return bool(strtobool(v))


def print_frame():
    callerframerecord = inspect.stack()[1]    # 0 represents this line
                                            # 1 represents line at caller
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    #   print(info.filename)                      # __FILE__     -> Test.py
    #   print(info.function)                      # __FUNCTION__ -> Main
    print("line {}:".format(info.lineno), end="")                       # __LINE__     -> 13