import os
import random
import pickle
from nltk.tokenize import word_tokenize 
import numpy as np
import pdb
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import networkx as nx
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import utils



class CEGDataset(Dataset):

    def __init__(self, data_dir=None, feats_dir=None, cache_dir=None, graph_path='', use_raw_qa=False, n_choice=None, shuffle=True, text_only=False):
        super(CEGDataset, self).__init__()

        random.seed(2021)
        self.raw_qa = dict()
        self.flattened_qa = dict()
        self.agg_qa = dict()
        self.scene_idcs = dict()
        
        if not hasattr(self, 'n_choice'):
            self.n_choice = n_choice
        if self.n_choice is None:
            self.n_choice = 1
        if not hasattr(self, 'shuffle'):
            self.shuffle = shuffle
        graphs_it6 = pickle.load(open(graph_path, 'rb'))
        self.split_ceg_data(graphs_it6)
        for phase in ["cegv3_core", "cegv3_core_train", "cegv3_core_validation", "cegv3_core_test"]:
            self.flattened_qa[phase], self.agg_qa[phase] = self.convert_ceg_new(phase, graphs_it6, n_choice=self.n_choice)
        if use_raw_qa:
            for phase in ["train", "validation"]:
                raw_qa = utils.load_json(os.path.join(data_dir, "{}.json".format(phase)))
                self.exclude_ceg_videos(phase, raw_qa)
                self.flattened_qa[phase], self.agg_qa[phase] = self.filter_qa(phase)

            self.add_single()
            self.add_small()

        self.phases = list(self.flattened_qa.keys())
        for phase in self.phases:
            print('{}: {} videos, {} questions, {} QA pairs'.format(phase, len(self.scene_idcs[phase]), len(self.agg_qa[phase]), len(self.flattened_qa[phase])))

        self.feats_dir = feats_dir

        print('load feats from %s' % (self.feats_dir))
    
    def split_ceg_data(self, ceg):
        scene_idcs = list(ceg.keys())
        if self.shuffle:
            random.shuffle(scene_idcs)
        num_scenes = len(scene_idcs)
        self.scene_idcs["cegv3_core_train"] = scene_idcs[:int(num_scenes * 0.8)]
        self.scene_idcs["cegv3_core_validation"] = scene_idcs[int(num_scenes * 0.8):int(num_scenes * 1)]
        self.scene_idcs["cegv3_core_test"] = []

    def exclude_ceg_videos(self, phase, raw_qa):
        cleaned_raw_qa = []
        scene_idcs = []
        for scene in raw_qa:
            if scene["scene_index"] not in self.scene_idcs["cegv3_core"]:
                cleaned_raw_qa.append(scene)
                scene_idcs.append(scene["scene_index"])
        self.scene_idcs[phase] = scene_idcs
        self.raw_qa[phase] = cleaned_raw_qa

    def filter_qa(self, phase):
        flattened = []
        agg = {}
        for scene in self.raw_qa[phase]:
            for question in scene["questions"]:
                if question["question_type"] == "explanatory":
                    if phase != "test":
                        answers = []
                        labels = []
                        for choice in question["choices"]:
                            answer = choice["choice"]
                        
                            label = 1 if choice["answer"] == "correct" else 0
                            answers.append(answer)
                            labels.append(label)
                        # if answers:
                            ques_idx = "{}_{}".format(scene["scene_index"], question["question_id"])
                            # for answer in answers:
                            flattened.append({"scene_index": scene["scene_index"], "question": question["question"], "ques_idx": ques_idx, "answer": answer, "video_filename": scene["video_filename"], "label": label})
                        agg[ques_idx] = {"scene_index": scene["scene_index"], "ques_idx": ques_idx, "question": question["question"], "video_filename": scene["video_filename"], "all_answers": answers, "labels": labels}
                    else:
                        flattened.append({"scene_index": scene["scene_index"], "question": question["question"]})
        return flattened, agg

    def convert_ceg(self, phase, graphs):
        if phase not in self.scene_idcs:
            scene_idcs = list(graphs.keys())
            random.shuffle(scene_idcs)
            self.scene_idcs[phase] = scene_idcs
        flattened = []
        agg = {}
        num_nodes = 0
        num_edges = 0
        for scene_index in self.scene_idcs[phase]:
            G = graphs[scene_index]["event_graph"]
            num_nodes += len(G.nodes)
            num_edges += len(G.edges)
            for i, n in enumerate(G.nodes):
                nbrs = list(G.predecessors(n))
                if len(nbrs) >= 4:
                    ques_idx = "{}_ceg{}".format(scene_index, i)
                    question = "Why " + n + "?"
                    num_correct = random.randrange(5)
                    random.shuffle(nbrs)
                    other_events = list(set(G.nodes) - set(nbrs))
                    random.shuffle(other_events)
                    
                    all_answers = nbrs[:num_correct] + other_events[:4 - num_correct]
                    labels = [1] * num_correct + [0] * (4 - num_correct)
                    for answer, label in zip(all_answers, labels):
                        flattened.append({"scene_index": scene_index, "question": question, "ques_idx": ques_idx, "answer": answer, "label": label})
                    agg[ques_idx] = {"scene_index": scene_index, "video_filename": graphs[scene_index]["video_filename"], "ques_idx": ques_idx, "question": question, "all_answers": all_answers, "labels": labels}
        print("phase: {}, num_nodes: {}, num_edges: {}".format(phase, num_nodes, num_edges))
        return flattened, agg
    
    def convert_ceg_w_neg(self, phase, graphs):
        if phase not in self.scene_idcs:
            scene_idcs = list(graphs.keys())
            random.shuffle(scene_idcs)
            self.scene_idcs[phase] = scene_idcs
        flattened = []
        agg = {}
        num_nodes = 0
        num_edges = 0
        for scene_index in self.scene_idcs[phase]:
            
            G = graphs[scene_index]["event_graph"]
            G_neg = graphs[scene_index]["event_graph_neg"]
            num_nodes += len(G.nodes)
            num_edges += len(G.edges)
           
            for i, n in enumerate(G.nodes):

                nbrs = list(G.predecessors(n))
                if len(nbrs) >= 4:
                    ques_idx = "{}_ceg{}".format(scene_index, i)
                    question = "Why " + n + "?"
                    num_correct = random.randrange(5)
                    random.shuffle(nbrs)
                    
                    #other_events = list(set(G.nodes) - set(nbrs))
                    
                    other_events = list(G_neg.predecessors(n))
                    if len(other_events) < 4:
                        print(scene_index, n, other_events)
                    random.shuffle(other_events)
                    
                   
                    all_answers = nbrs[:num_correct] + other_events[:4 - num_correct]
                    labels = [1] * num_correct + [0] * (4 - num_correct)
                    for answer, label in zip(all_answers, labels):
                        flattened.append({"scene_index": scene_index, "question": question, "ques_idx": ques_idx, "answer": answer, "label": label})
                    agg[ques_idx] = {"scene_index": scene_index, "video_filename": graphs[scene_index]["video_filename"], "ques_idx": ques_idx, "question": question, "all_answers": all_answers, "labels": labels}
        print("phase: {}, num_nodes: {}, num_edges: {}".format(phase, num_nodes, num_edges))
        return flattened, agg
    
    def convert_ceg_new_filter(self, phase, graphs):
        # for filtered graph, not neg and pos graph
        # for clevrer_humans
        if phase not in self.scene_idcs:
            scene_idcs = list(graphs.keys())
            random.shuffle(scene_idcs)
            self.scene_idcs[phase] = scene_idcs
        flattened = []
        agg = {}
        num_nodes = 0
        num_edges = 0
        for scene_index in self.scene_idcs[phase]:
            G = graphs[scene_index]["CEG_filtered"]
            num_nodes += len(G.nodes)
            num_edges += len(G.edges)
            for i, n in enumerate(G.nodes):
                nbrs = list(G.predecessors(n))
                num_correct = random.randrange(5)
                if len(nbrs) >= num_correct and len(G.nodes) >= 4:
                    ques_idx = "{}_ceg{}".format(scene_index, i)
                    question = "Which of the following is responsible for " + n + "?"
                    random.shuffle(nbrs)
                    other_events = list(set(G.nodes) - set(nbrs))
                    random.shuffle(other_events)
                    
                    all_answers = nbrs[:num_correct] + other_events[:4 - num_correct]
                    if len(all_answers) != 4:
                        continue
                    labels = [1] * num_correct + [0] * (4 - num_correct)
                    for answer, label in zip(all_answers, labels):
                        flattened.append({"scene_index": scene_index, "question": question, "ques_idx": ques_idx, "answer": answer, "label": label})
                    agg[ques_idx] = {"scene_index": scene_index, "video_filename": graphs[scene_index]["video_filename"], "ques_idx": ques_idx, "question": question, "all_answers": all_answers, "labels": labels}
        print("phase: {}, num_nodes: {}, num_edges: {}".format(phase, num_nodes, num_edges))
        return flattened, agg
    
    def convert_ceg_new(self, phase, graphs, n_choice=1):
        # for clevrer_humans
        if phase not in self.scene_idcs:
            scene_idcs = list(graphs.keys())
            if self.shuffle:
                random.shuffle(scene_idcs)
            self.scene_idcs[phase] = scene_idcs
        flattened = []
        agg = {}
        num_nodes = 0
        num_edges = 0
        n_pos = 0
        n_neg = 0
        for scene_index in self.scene_idcs[phase]:
            G = graphs[scene_index]["CEG_full"]
            G_pos = graphs[scene_index]["CEG_pos"]
            G_neg = graphs[scene_index]["CEG_neg"]
            num_nodes += len(G.nodes)
            num_edges += len(G.edges)
            
            for i, n in enumerate(G_pos.nodes):
                nbrs = list(G_pos.predecessors(n))
                nbrs_neg = list(G_neg.predecessors(n))
                #num_correct = random.randrange(5)
                
                num_correct = n_choice
                if len(nbrs) >= num_correct and len(G.nodes) >= n_choice: #and len(G.edges) >= 2:#len(nbrs_neg) >= 4 - num_correct:
                    num_correct = random.randrange(n_choice+1)
                    ques_idx = "{}_ceg{}".format(scene_index, i)
                 
                    question = "Which of the following is responsible for " + n + "?"
                    if self.shuffle:
                        random.shuffle(nbrs)
                    random.shuffle(nbrs_neg)
                    other_events = list(set(G.nodes) - set(nbrs) - set(nbrs_neg))
                    if self.shuffle:
                        random.shuffle(other_events)
                    other_events = nbrs_neg + other_events
                    all_answers = nbrs[:num_correct] + other_events[:n_choice - num_correct]
                    if len(all_answers) != n_choice:
                        continue
                    n_pos += num_correct
                    n_neg += n_choice - num_correct
                    labels = [1] * num_correct + [0] * (n_choice - num_correct)
            
                    for answer, label in zip(all_answers, labels):
                        flattened.append({"scene_index": scene_index, "question": question, "ques_idx": ques_idx, "answer": answer, "label": label})
                    agg[ques_idx] = {"scene_index": scene_index, "video_filename": graphs[scene_index]["video_filename"], "ques_idx": ques_idx, "question": question, "all_answers": all_answers, "labels": labels}
        
        print("phase: {}, num_nodes: {}, num_edges: {}".format(phase, num_nodes, num_edges))
        print('positive, negative choices', n_pos, n_neg)
        return flattened, agg



    def add_single(self):
        # self.phases.append("train_single")
        self.scene_idcs["train_single"] = [0]
        single_ques_idx = "0_11"
        single_qa = []
        for example in self.flattened_qa["train"]:
            if example["ques_idx"] == single_ques_idx:
                single_qa.append(example)
        self.flattened_qa["train_single"] = single_qa
        self.agg_qa["train_single"] = {single_ques_idx: self.agg_qa["train"][single_ques_idx]}

    def add_small(self):
        # self.phases.append("train_small")
        self.scene_idcs["train_small"] = list(range(1000))
        self.flattened_qa["train_small"] = []
        self.agg_qa["train_small"] = {}
        for example in self.flattened_qa["train"]:
            ques_idx = example["ques_idx"]
            if int(ques_idx.split("_")[0]) < 1000:
                self.flattened_qa["train_small"].append(example)
                self.agg_qa["train_small"][ques_idx] = self.agg_qa["train"][ques_idx]


class CEGGloveDataset(CEGDataset):
    def __init__(self, graph_path=None, cur_phase = 'cegv3_core_train', data_dir=None, feats_dir=None, cache_dir=None, dim_word=300, ques_max_len=32, ans_max_len=32, text_only=False, n_choice=1, shuffle=True):
        super(CEGGloveDataset, self).__init__(graph_path=graph_path, data_dir=data_dir, feats_dir=feats_dir, cache_dir=cache_dir, n_choice=n_choice, shuffle=shuffle, text_only=text_only)
        self.q_max_len = ques_max_len
        self.a_max_len = ans_max_len
        self.shuffle = shuffle
        #cur_phase = 'cegv3_core_train'
        assert cur_phase in self.phases
        self.set_phase(cur_phase)
        self.text_only = text_only
        
        # set word embedding / vocabulary
        self.glove_embedding_path = os.path.join(os.path.dirname(data_dir), "glove/glove.6B.300d.txt")
        self.word2idx_path = os.path.join(cache_dir, "word2idx.pickle")
        self.idx2word_path = os.path.join(cache_dir, "idx2word.pickle")
        self.vocab_embedding_path = os.path.join(cache_dir, "vocab_embedding.pickle")
        self.embedding_dim = dim_word
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3} # The idx of "<pad>" must be 0
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.offset = len(self.word2idx)

        # build/load vocabulary
        if not utils.files_exist([self.word2idx_path, self.idx2word_path, self.vocab_embedding_path]):
            print("\nNo cache founded.")
            self.build_word_vocabulary()
        else:
            print("\nLoading cache ...")
            self.word2idx = utils.load_pickle(self.word2idx_path)
            self.idx2word = utils.load_pickle(self.idx2word_path)
            self.vocab_embedding = utils.load_pickle(self.vocab_embedding_path)

    def set_phase(self, phase):
        self.phase = phase
        self.cur_qa = self.flattened_qa[phase]
        self.gt = self.agg_qa[phase]

    @staticmethod
    def load_glove(filename):
        """ Load glove embeddings into a python dict
        returns { word (str) : vector_embedding (torch.FloatTensor) }"""
        glove = {}
        with open(filename) as f:
            for line in f.readlines():
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def build_word_vocabulary(self, word_count_threshold=0):
        """borrowed this implementation from @karpathy's neuraltalk."""
        print("Building word vocabulary starts.\n")
        all_sentences = []
        for phase in self.phases:
            if "ceg" not in phase:
                for scene in self.raw_qa[phase]:
                    for question in scene["questions"]:
                        all_sentences.append(question["question"])
                        if question["question_type"] == "descriptive":
                            if phase != "test":
                                all_sentences.append(question["answer"])
                        else:
                            for choice in question["choices"]:
                                all_sentences.append(choice["choice"])
            else:
                for qa_example in self.flattened_qa[phase]:
                    all_sentences.extend([qa_example["question"], qa_example["answer"]])

        word_counts = {}
        for sentence in tqdm(all_sentences):
            for w in self.line_to_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.word2idx.keys()]
        print("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" %
              (len(vocab), word_count_threshold))

        # build index and vocabularies
        for idx, w in enumerate(vocab):
            self.word2idx[w] = idx + self.offset
            self.idx2word[idx + self.offset] = w
        print("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word)))

        # Make glove embedding.
        print("Loading glove embedding at path : %s\n" % self.glove_embedding_path)
        glove_full = self.load_glove(self.glove_embedding_path)
        print("Glove Loaded, building word2idx, idx2word mapping. This may take a while.\n")
        glove_matrix = np.zeros([len(self.idx2word), self.embedding_dim])
        glove_keys = glove_full.keys()
        for i in tqdm(range(len(self.idx2word))):
            w = self.idx2word[i]
            w_embed = glove_full[w] if w in glove_keys else np.random.randn(self.embedding_dim) * 0.4
            glove_matrix[i, :] = w_embed
        self.vocab_embedding = glove_matrix
        print("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word)))
        print("Vocab embedding size is :", glove_matrix.shape)

        print("Saving cache files ...\n")
        utils.save_pickle(self.word2idx, self.word2idx_path)
        utils.save_pickle(self.idx2word, self.idx2word_path)
        utils.save_pickle(glove_matrix, self.vocab_embedding_path)
        print("Building  vocabulary done.\n")

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        example = self.cur_qa[ix]
        scene_index = example["scene_index"]
        # Video features
        fc_feat = []
        if self.text_only:
            fc_feat = np.zeros((0))
        else:
            for feats_dir in self.feats_dir:
                #fc_feat.append(np.load(os.path.join(feats_dir, 'video_{:05d}.npy'.format(scene_index))))
                fc_feat.append(np.load(os.path.join(feats_dir, 'video_{}.npy'.format(scene_index))))
            fc_feat = np.concatenate(fc_feat, axis=1)
            # if self.with_c3d == 1:
            #     c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video_{:05d}.npy'.format(scene_index)))
            #     c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
            #     fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)

            # Questions and answers
        question_arr = np.array(self.numericalize(example["question"], max_len=self.q_max_len, eos=False))
        question_mask = (question_arr > 0).astype(int)
        if self.phase != "test":
            answer_arr = np.array(self.numericalize(example["answer"], max_len=self.a_max_len, eos=True))
            answer_mask = (answer_arr > 0).astype(int)

        data = {}
        data['fc_feats'] = fc_feat
        data['question'] = question_arr
        data['question_mask'] = question_mask
        data['question_len'] = np.sum(question_mask)
        data['scene_index'] = scene_index
        data["ques_idx"] = example["ques_idx"]
        data["label"] = example["label"]
        if self.phase != "test":
            data['answer'] = answer_arr
            data['answer_mask'] = answer_mask
            data['answer_len'] = np.sum(answer_mask)
        return data

    @staticmethod
    def line_to_words(line):
        line = line.lower()
        words = word_tokenize(line)
        return words

    def numericalize(self, sentence, max_len, eos=True):
        """convert words to indices"""
        words = self.line_to_words(sentence)
        words = self.truncate_or_pad(words, max_len, eos)
        sentence_indices = []
        for w in words:
            if w in self.word2idx:
                sentence_indices.append(self.word2idx[w])
            else:
                sentence_indices.append(self.word2idx["<unk>"])
        return sentence_indices

    @staticmethod
    def truncate_or_pad(words, max_len, eos):
        if eos:
            if len(words) < max_len - 2:
                words = ["<sos>"] + words + ["<eos>"]
                words += ["<pad>"] * (max_len - len(words))
            else:
                words = words[:max_len - 2]
                words = ["<sos>"] + words + ["<eos>"]
        elif len(words) < max_len:
            words += ["<pad>"] * (max_len - len(words))
        else:
            words = words[:max_len]
        return words

    def __len__(self):
        return len(self.cur_qa)

    def get_vocab_size(self):
        return len(self.word2idx)

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word

    def get_offset(self):
        return self.offset

if __name__ == "__main__":
    # opt = opts.parse_opt()
    # opt = vars(opt)
    data_dir = "/viscam/data/clevrer"
    cache_dir = "cache"
    feats_dir = ["cache/feats/resnet50"]

    dataset = CEGGloveDataset(data_dir=data_dir, cache_dir=cache_dir, feats_dir=feats_dir)
    dataloader = DataLoader(dataset, batch_size=4)
    for data in dataloader:
        fc_feats = data['fc_feats'].cuda()
        question = data['question'].cuda()
        question_mask = data['question_mask'].cuda()
        break