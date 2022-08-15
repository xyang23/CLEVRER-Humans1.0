#!/usr/bin/env python
# coding: utf-8

import glob
import json
import os
import pickle
import numpy as np
import pdb

import csv, sys
sys.path.append(".")

map_color = {'gray' : 0, 'red' : 1, 'blue' : 2, 'green': 3, 'brown': 4, 'cyan' : 5, 'purple' : 6, 'yellow' : 7, 'gold': 7, 'silver': 0};
map_shape = {'cube': 0, 'sphere': 1, 'cylinder': 2, 'ball': 1}
map_material = {'metal': 0, 'rubber': 1, 'magenta': 0}


#  Generate data for single object trajectory

def get_single_object_trajectory(obj, motion_trajectory):
    trajectory = []
    for frame in motion_trajectory: # for each of the 128 frames
        obj_index = obj['object_id']
        obj_property = [0] * (len(set(map_color.values())) + len(set(map_shape.values())))
        obj_property[map_color[obj['color']]] = 1
        obj_property[map_shape[obj['shape']] + len(set(map_color.values()))] = 1
        trajectory = trajectory +  [int(frame['objects'][obj_index]['inside_camera_view'])] + frame['objects'][obj_index]['location']+ frame['objects'][obj_index]['orientation'] + frame['objects'][obj_index]['velocity'] + frame['objects'][obj_index]['angular_velocity']
        trajectory += obj_property
    return trajectory

data = {}

for folder in sorted(glob.glob('/viscam/u/xyang23/annotation_1*')):
    for file in sorted(os.listdir(folder)):
        if file[0] == '.':
            continue
        if int(file[11: 16]) > 10599: # 1199:# 499:
            break
       # if int(file[11: 16]) not in ind:
       #     continue
        with open(os.path.join(folder, file)) as f:
            tmp = json.load(f)
        data[file[11: 16]] = {}
        data[file[11: 16]]['single_object'] = []
        for obj in tmp['object_property']:
            trajectory = get_single_object_trajectory(obj, tmp['motion_trajectory'])    
            data[file[11: 16]]['single_object'].append({'object': obj, 'trajectory': trajectory})

with open('/viscam/u/xyang23/lstm/checkpoints/CLEVRER_10000-10599_single_traj.p', 'wb') as f: 
    pickle.dump(data, f)

# generate data for double object
def get_double_object_trajectory(obj1, obj2, motion_trajectory):
    trajectory = []
    for frame in motion_trajectory: # for each of the 128 frames
        for obj in [obj1, obj2]:
            obj_index = obj['object_id']
            obj_property = [0] * (len(set(map_color.values())) + len(set(map_shape.values())))
            obj_property[map_color[obj['color']]] = 1
            obj_property[map_shape[obj['shape']] + len(set(map_color.values()))] = 1
            trajectory = trajectory +  [int(frame['objects'][obj_index]['inside_camera_view'])] + frame['objects'][obj_index]['location']+ frame['objects'][obj_index]['orientation'] + frame['objects'][obj_index]['velocity'] + frame['objects'][obj_index]['angular_velocity']
            trajectory += obj_property
    return trajectory

with open('/viscam/u/xyang23/lstm/checkpoints/CLEVRER_10000-10599_single_traj.p', 'rb') as f: 
    data = pickle.load(f)
for t, folder in enumerate(sorted(glob.glob('/viscam/u/xyang23/annotation_1*'))):
    for file in sorted(os.listdir(folder)):
        if file[0] == '.':
            continue
        if int(file[11: 16]) > 10599: #1199:# 499:
            break  
        #data[str(file[11: 16])] = {}
        data[str(file[11: 16])]['double_object'] = []
        with open(os.path.join(folder, file)) as f:
            tmp = json.load(f)
            n_obj = len(tmp['object_property'])
            for id1 in range(n_obj):
                obj1 = tmp['object_property'][id1]
                #trajectory1 = get_single_object_trajectory(obj1, tmp['motion_trajectory'])
                for id2 in range(id1 + 1, n_obj):
                    obj2 = tmp['object_property'][id2]
                    trajectory = get_double_object_trajectory(obj1, obj2, tmp['motion_trajectory'])
                    data[str(file[11: 16])]['double_object'].append({"objects": [obj1, obj2], "trajectory": trajectory})
with open('/viscam/u/xyang23/lstm/checkpoints/CLEVRER_10000-10599_single_double_traj.p', 'wb') as f: 
    pickle.dump(data, f)

def is_close(obj1, obj2):
    loc_diff = np.absolute(np.array(obj1['location']) - np.array(obj2['location'])) < np.array([0.5, 0.5, 0.5])
   # print(obj1['location'], obj2['location'], np.array(obj1['location']) - np.array(obj2['location']), loc_diff)
    return loc_diff.all()

def is_reverse_direction(obj, close_event, ann_content):
    is_reverse = False
    cur_location = np.array(obj['location'])
    cur_orientation =  np.array(obj['orientation'])
    cur_velocity = np.array(obj['velocity'])
    cur_angular = np.array(obj['angular_velocity'])
    
    for i in range(close_event["start"], close_event["end"]):
        frame = ann_content['motion_trajectory'][i]
        for cur_obj in frame['objects']:
            if cur_obj['object_id'] == obj['object_id']:
                if i > close_event["start"]:
                    if (cur_obj['velocity'][:-1] * cur_velocity[:-1] < 0).any():
                        is_reverse = True
                
                cur_location = np.array(cur_obj['location'])
                cur_orientation =  np.array(cur_obj['orientation'])
                cur_velocity = np.array(cur_obj['velocity'])
                cur_angular = np.array(cur_obj['angular_velocity'])
    return is_reverse

def is_angular_change(obj, close_event, ann_content):
    is_angular = 0
    cur_location = np.array(obj['location'])
    cur_orientation =  np.array(obj['orientation'])
    cur_velocity = np.array(obj['velocity'])
    cur_angular = np.array(obj['angular_velocity'])
    
    for i in range(close_event["start"], close_event["end"]):
        frame = ann_content['motion_trajectory'][i]
        for cur_obj in frame['objects']:
            if cur_obj['object_id'] == obj['object_id']:
                if i > close_event["start"]:
                    if (np.absolute(np.array(cur_obj['angular_velocity']) - cur_angular) > np.array([2, 2, 2])).any():
                        if (np.absolute(np.array(cur_obj['angular_velocity'])) - np.absolute(cur_angular) > np.array([2, 2, 2])).any():
                            is_angular = 1
                        else:
                            is_angular = -1
                cur_location = np.array(cur_obj['location'])
                cur_orientation =  np.array(cur_obj['orientation'])
                cur_velocity = np.array(cur_obj['velocity'])
                cur_angular = np.array(cur_obj['angular_velocity'])
    return is_angular

    
    
def get_event(ann_content, obj1_id, obj2_id, event='collision'):
    # get the events
    close_frames = []
    
    for frame in ann_content['motion_trajectory']:
       # print(frame['frame_id'])
        for obj in frame['objects']: 
            
            # get the object info in the frame
            if obj1_id == obj['object_id']:
                obj1 = obj
            if obj2_id == obj['object_id']:
                obj2 = obj
        if is_close(obj1, obj2):
            close_frames.append(frame['frame_id'])
            #print(frame['frame_id'], ann_content['object_property'][obj1_id], ann_content['object_property'][obj2_id])
    n_frame = len(close_frames) 
    if n_frame == 0:
        return close_frames # return empty list     
    event_list = []
    start_frame = 0   
  #  print(ann_content['object_property'][obj1_id], ann_content['object_property'][obj2_id])
    for i in range(1, n_frame):       
        if close_frames[i] - close_frames[i - 1] - 1 > 0:
            event_list.append({"start": close_frames[start_frame], "end": close_frames[i - 1], "event": "close"})
            start_frame = i           
    event_list.append({"start": close_frames[start_frame], "end": close_frames[n_frame - 1], "event": "close"}) # last frame
    event_list_copy = event_list.copy()
    
    for close_event in event_list_copy:
        if close_event['end'] - close_event['start'] > 20: # more than 20 frame close = relatively still = move together
            event_type = "move_together"            
        else:
            event_type = "collide"
            if (is_reverse_direction(obj1, close_event, ann_content) or is_reverse_direction(obj2, close_event, ann_content)):
                # todo: add one object should at least move
                event_type += "-bounce_back"
            else:
                event_type += "-push"
            if is_angular_change(obj1, close_event, ann_content) > 0:
                event_type += "-obj1_get_spin" 
            if is_angular_change(obj1, close_event, ann_content) < 0:
                event_type += "-obj1_stop_spin"
               
            if is_angular_change(obj2, close_event, ann_content) > 0:
                event_type += "-obj2_get_spin" 
            if is_angular_change(obj2, close_event, ann_content) < 0:
                event_type += "-obj2_stop_spin"

        event_list.append({"start": close_frames[start_frame], "end": close_frames[n_frame - 1], "event": event_type})
            
    return event_list
 

with open('/viscam/u/xyang23/lstm/checkpoints/CLEVRER_10000-10599_single_double_traj.p', 'rb') as f: 
    data = pickle.load(f)
for folder in sorted(glob.glob('/viscam/u/xyang23/annotation_1*')):
    for file in sorted(os.listdir(folder)):
        if file[0] == '.':
            continue
        if int(file[11: 16]) > 10599: #1199:# 499:
            break  
        with open(os.path.join(folder, file)) as f:
            tmp = json.load(f)
        data[file[11:16]]['manual_events'] = []
        n_obj = len(tmp['object_property'])
        for id1 in range(n_obj):
            for id2 in range(id1 + 1, n_obj):
                events = get_event(tmp, id1, id2)
                if len(events) > 0:
        # print(len(tmp['motion_trajectory'])) = 128 frames
                    obj_pair = {'obj1': tmp['object_property'][id1], 'obj2': tmp['object_property'][id2], 'event': events}
                    data[file[11:16]]['manual_events'].append(obj_pair)

with open('/viscam/u/xyang23/lstm/checkpoints/CLEVRER_10000-10599_single_double_traj_w_manual.p', 'wb') as f: 
    pickle.dump(data, f)


# Generating bounding box videos

def transform_loc_sim(x, y, fov=76):
    scale = 2.2
    return int(y*scale*1.45), int(x*scale*1.45)


def get_loc(frame, obj_id):
    if True:
        loc = frame[obj_id]['location']
        b_width = 25
        b_height = 28

        new_x, new_y = transform_loc(x=loc[1], y=loc[0])

        obj = [(new_x - b_width, new_y - b_height), 
               (new_x + b_width, new_y + b_height)]
    else:
        obj = [(0, 0), (0, 0)]
    return obj

def is_matched(obja, objb):
    return obja['shape'] == objb['shape'] and obja['material'] == objb['material'] and obja['color'] == objb['color']

def get_loc_sim(frame, ori_obj):

    obj = [(0, 0), (0, 0), (0, 0)]
 
    for cur_obj in frame['objects']:
        if is_matched(cur_obj, ori_obj):
            x, y = cur_obj['x'], cur_obj['y']
            b_width = 25
            b_height = 28
            new_x, new_y = transform_loc_sim(x, y)
            obj = [(new_x - b_width, new_y - b_height), 
                   (new_x + b_width, new_y + b_height), (new_x - 10, new_y + 10)]
    return obj

def get_loc_sim_interpolate(frame1, frame2, mod, ori_obj):
    obj = [(0, 0), (0, 0), (0, 0)]
    for cur_obj in frame1['objects']:
        for cur_obj_2 in frame2['objects']:
            if is_matched(cur_obj, ori_obj) and is_matched(cur_obj_2, ori_obj):
                x, y = cur_obj['x'], cur_obj['y']
                x2, y2 = cur_obj_2['x'], cur_obj_2['y']
                x = x + (x2 - x) / 5 * mod
                y = y + (y2 - y) / 5 * mod
                b_width = 25
                b_height = 28
                new_x, new_y = transform_loc_sim(x, y)

                obj = [(new_x - b_width, new_y - b_height), 
                       (new_x + b_width, new_y + b_height), (new_x - 10, new_y + 10)]

    return obj

from typing import List, Tuple
import cv2
import pdb

def draw_bounding_box(video_in: str, video_out: str, bounding: List[List[List[Tuple[int]]]]):
    # tuple: (x, y); list of tuple: 2 tuples, upper left, lower right; list 
# modified to enable variated lengths of number of objs
    cap = cv2.VideoCapture(video_in)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            for obj in bounding[i]:
                cv2.rectangle(frame, obj[0], obj[1], obj[3], obj[4])  # green box
            writer.write(frame)
            i += 1
        else:
            break
    cap.release()
    writer.release()


def draw_bounding_box_new(video_in: str, video_out: str, bounding: List[List[List[Tuple[int]]]]):
    import imageio
    video = []
    # tuple: (x, y); list of tuple: 2 tuples, upper left, lower right; list 
# modified to enable variated lengths of number of objs
    cap = cv2.VideoCapture(video_in)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            for obj in bounding[i]:
                cv2.rectangle(frame, obj[0], obj[1], obj[3], obj[4])  # green box
            
            """for k in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    t = frame[k][j][0]
                    frame[k][j][0] = frame[k][j][2]
                    frame[k][j][2] = t"""
            frame[:, :, [2, 0]] = frame[:, :, [0, 2]]
            #writer.write(new_frame)
            video.append(list(frame))
            #writer.write(frame)
            i += 1
        else:
            break
    cap.release()
    #writer.release() 
    imageio.mimsave(video_out, np.asarray(video), fps=fps)
    

with open('/viscam/u/xyang23/lstm/checkpoints/CLEVRER_10000-10599_5_29_step3.p', 'rb') as f: 
    visualize_data = pickle.load(f)
video_cnt = 0    
for video_idx, video in enumerate(visualize_data.keys()):
    #if int(video) > 10100:
    #    continue
    with open("/viscam/u/xyang23/CLEVRER/executor/data/propnet_preds/with_edge_supervision_old/sim_"+ str(video) +".json") as f:
        tmp = json.load(f)
        tmp = tmp['predictions'][0]
    print(video)
    visualize_data[video]['bounding'] = {}
    for pair in visualize_data[video]['pair_index']:
        bounding = []
        for i in range(128):
            cur_bounding = [] #[obj1a, obj2a, obj1b, obj2b]
            for obj in visualize_data[video]['objects'][pair[0]]:
                obj_loc = get_loc_sim_interpolate(tmp['trajectory'][i//5], tmp['trajectory'][i//5 + 1], i%5, obj)
                obj_loc.append((0, 255, 0))
                obj_loc.append((4))
                cur_bounding.append(obj_loc)
            for obj in visualize_data[video]['objects'][pair[1]]:
                obj_loc = get_loc_sim_interpolate(tmp['trajectory'][i//5], tmp['trajectory'][i//5 + 1], i%5, obj)
                obj_loc.append((0, 0, 255))
                obj_loc.append((2))
                cur_bounding.append(obj_loc)
            bounding.append(cur_bounding)            
        video_in = '/viscam/data/clevrer/video_all/video_' + video + '.mp4'
        video_out_dir = '/viscam/u/xyang23/lstm/bounding_box_data/bbox_video_10000-10599_5_29'
        if not os.path.exists(video_out_dir):
            os.mkdir(video_out_dir)
            print('New directory created at', video_out_dir)
        video_out = video_out_dir + '/video_'+ video + '_' + str(pair[0]) + '_'+ str(pair[1]) +'.mp4'
        video_cnt += 1
        draw_bounding_box_new(video_in, video_out, bounding)
        
  




