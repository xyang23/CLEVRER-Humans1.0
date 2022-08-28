# for human annotating CLEVRER sentences 

import pdb
import numpy as np
import json
import pickle
import csv
import random
import shutil, os
import networkx as nx
import sys

def get_video(e):
    # get the video name
    url = e['video_url_bbox']
    video_name = url.split('/')[-1]
    return '{:05}'.format(int(video_name.split('_')[1][:5]))


def get_question(s):
    # get the question by deleting what is responsible for ... 
    return s.split('for ')[1][:-1]


# flatten human answer
answer_path =  "causal_diff_data/human_valid_q.p"

with open(answer_path, 'rb') as f: 
    ans = pickle.load(f)

ans.sort(key=get_video)
ans_flatten = {}
for pair in ans:
    video = str(get_video(pair))
    event_A = pair['event_A']
    event_B = pair['event_B']  
    if video not in ans_flatten.keys():
        ans_flatten[video] = {'CEG_full': nx.DiGraph(), 'CEG_pos': nx.DiGraph(), 'CEG_neg':nx.DiGraph()}
    ans_flatten[video]['CEG_full'].add_weighted_edges_from([(event_A, event_B, int(pair['causal_level']))])

# generate CEG from human answer
causal_threshold =  4 
ratio_list = []
for video in ans_flatten.keys():
    ans_flatten[video]['video_filename'] = 'video_' + video +'.mp4'
    ans_flatten[video]['CEG_pos'] = copy.deepcopy(ans_flatten[video]['CEG_full'])
    ans_flatten[video]['CEG_neg'] = copy.deepcopy(ans_flatten[video]['CEG_full'])
    for sentence  in ans_flatten[video]['CEG_full'].nodes():
        all_nodes = ans_flatten[video]['CEG_full'].nodes()
        for v in all_nodes:
            if ans_flatten[video]['CEG_full'].has_edge(sentence, v):
                weight = ans_flatten[video]['CEG_full'].get_edge_data(sentence, v)['weight']
                if weight < causal_threshold:
                    ans_flatten[video]['CEG_pos'].remove_edge(sentence, v)
                else:
                    ans_flatten[video]['CEG_neg'].remove_edge(sentence, v)

question_path = 'causal_diff_data/counterfactual_valid_q.json'
with open(question_path, 'rb') as f:
    questions_counter = json.load(f)

question_path = 'causal_diff_data/clevrer_valid_q.json'
with open(question_path, 'rb') as f:
    questions_clevrer = json.load(f)

idx_list = list(ans_flatten.keys())

stats = {'total_choices':0,'total_questions':0, 'human_heur_true_cnt':0,
         'heur_true_cnt':0, 'human_counter_true_cnt':0,'counter_true_cnt':0,'human_true_cnt':0}

for idx in idx_list:
    ann_counter = questions_counter[str(idx)]
    ann_clevrer = questions_clevrer[str(idx)]
    video_idx = str(int(idx))

    for q_idx, question in enumerate(ann_clevrer['questions']):
        if question['question_type'] != 'explanatory':
            continue
        
        neg_flag = False # not responsible for
        if 'not' in question['question']:
            # ignore repeated but negated questions
            continue
            neg_flag = True
        else:
            pass
            #continue
        stats['total_questions'] += 1
        sentence1 = get_question(question['question'])
        for c_idx, choice in enumerate(question['choices']):
            sentence2 = choice['choice']
            stats['total_choices'] += 1
            # get clevrer heuristic answer
            if choice['answer'] == 'correct':
                heur_ans = True
            else:
                heur_ans = False
            # get counterfactual answer
            if ann_counter['questions'][q_idx]['choices'][c_idx]['answer'] == 'correct':
                counter_ans = True
            else:
                counter_ans = False

            human_ans = ans_flatten[video_idx]['CEG_pos'].has_edge(sentence1, sentence2)
            if neg_flag:
                heur_ans = not heur_ans
                counter_ans = not counter_ans
                human_ans = not human_ans

          
            if heur_ans and human_ans:
                stats['human_heur_true_cnt'] += 1
           
            if heur_ans:
                stats['heur_true_cnt'] += 1
                
            if human_ans:
                stats['human_true_cnt'] += 1 
                
            if human_ans and counter_ans:
                stats['human_counter_true_cnt'] += 1
           
            if counter_ans:
                stats['counter_true_cnt'] += 1
                
            if human_ans and (counter_ans and heur_ans):
                stats['human_and_true_cnt'] += 1
            if human_ans and (counter_ans or heur_ans):
                stats['human_or_true_cnt'] += 1
            
                
stats['human|heur'] = stats['human_heur_true_cnt']/stats['heur_true_cnt']
print('P(human|clevrer)', stats['human|heur'])
stats['heur|human'] = stats['human_heur_true_cnt']/stats['human_true_cnt']
print('P(clevrer|human)', stats['heur|human'])

stats['human|counter'] = stats['human_counter_true_cnt']/stats['counter_true_cnt']
print('P(human|counter)', stats['human|counter'])
stats['counter|human'] = stats['human_counter_true_cnt']/stats['human_true_cnt']
print('P(counter|human)', stats['counter|human'])


