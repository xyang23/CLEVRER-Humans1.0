import os
import json
import tqdm
import pdb
import sys
from tqdm import tqdm
import argparse

from executor_sim import Executor
from simulation import Simulation


parser = argparse.ArgumentParser()
parser.add_argument('--n_progs', default=1000)
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Use interaction network
#args = parser.parse_args()
args, unknown = parser.parse_known_args()

if args.use_event_ann != 0:
    
    raw_motion_dir = 'data/propnet_preds/with_edge_supervision_old'
else:
    raw_motion_dir = 'data/propnet_preds/without_edge_supervision'
if args.use_in != 0:
    raw_motion_dir = 'data/propnet_preds/interaction_network'


question_path = 'data/validation.json'
if args.n_progs == 'all':
    program_path = 'data/parsed_programs/mc_allq_allc.json'
else:
    program_path = 'data/parsed_programs/mc_{}q_{}c_val_new.json'.format(args.n_progs, int(args.n_progs)*4)


with open('data/validation.json') as f:
    anns = json.load(f)

q_num_total = 0
c_num_total = 0
#pred_map = {'yes': 'correct', 'no': 'wrong', 'error': 'error'}
pred_map = {'yes': 'correct', 'no': 'wrong', 'error': 'wrong'}
idx_list = []

for ann_idx in idx_list:

    question_scene = anns[ann_idx] 
    file_idx = ann_idx + 10000 
    ann_path = os.path.join(raw_motion_dir, 'sim_%05d.json' % file_idx)
    sim = Simulation(ann_path, use_event_ann=(args.use_event_ann != 0))
    exe_old = Executor(sim)
    exe_new = Executor(sim, anc_old=False)

    for q_idx, q in enumerate(question_scene['questions']): 
        question = q['question'] 
        q_type = q['question_type']       
        if not q_type.startswith('explanatory'):
            continue
        q_num_total += 1
        q_ann = anns[ann_idx]['questions'][q_idx]
        for c_idx, c in enumerate(q_ann['choices']):
            c_num_total += 1
            full_pg = c['program'] + q_ann['program']
            ans = c['answer'] # ans is original answer in clevrer
            pred_old = pred_map[exe_old.run(full_pg, debug=False)] # clevrer heuristic 
            pred_new = pred_map[exe_new.run(full_pg, debug=False)] # counterfactual intervention
            if pred_new != ans:
                anns[ann_idx]['questions'][q_idx]['choices'][c_idx]['answer'] = pred_new
                
        if correct_question:
            q_correct_num += 1
with open('counterfactual_valid_q_new.json', 'w') as f:
    json.dump(anns, f)









