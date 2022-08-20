import torch
import torch.nn as nn

import sys,os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        for param in self.model.parameters():
            param.requires_grad = False
        print("BERT init")

    def forward(self, question):
        inputs = self.tokenizer(question, return_tensors="pt", padding='max_length', max_length=32)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
