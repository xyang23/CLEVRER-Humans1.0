# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for running model on CLEVRER."""

import jacinle
import json
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import random
import model as modellib

BATCH_SIZE = 8
NUM_FRAMES = 25
NUM_OBJECTS = 8

_BASE_DIR = flags.DEFINE_string(
    "base_dir", "./clevrer_monet_latents",
    "Directory containing checkpoints and MONet latents.")

_SCENE_IDX = flags.DEFINE_integer(
    "scene_idx", 10494, "Scene index of CLEVRER video.") #changed


def load_monet_latents(base_dir, scene_index, phase='train'):
    filename = f"{base_dir}/{phase}/{scene_index}.npz"
    filename = f"{base_dir}/{phase}/{scene_index}.npz" # changed
    with open(filename, "rb") as f:
        return np.load(f)


def _split_string(s):
    """Splits string to words and standardize alphabet."""
    return s.lower().replace("?", "").split()


def _pad(array, length):
    """Pad an array to desired length."""
    if length - array.shape[0] < 0:
        return array[: length]
    return np.pad(array, [(0, length - array.shape[0])] + [(0, 0) for _ in range(len(array.shape) - 1)], mode="constant")


def encode_sentence(token_map, sentence, pad_length):
    """Encode CLEVRER question/choice sentences as sequence of token ids."""
    ret = np.array([token_map["question_vocab"].get(w, 0) for w in _split_string(sentence)], np.int32)
    return _pad(ret, pad_length)


def encode_choices(token_map, choices):
    """Encode CLEVRER choices."""
    arrays = [encode_sentence(token_map, choice["choice"], modellib.MAX_CHOICE_LENGTH) for choice in choices]
    ret = _pad(np.stack(arrays, axis=0), modellib.NUM_CHOICES)
    return ret


def main(unused_argv):
    base_dir = _BASE_DIR.value
    #  with open(f"{base_dir}/vocab.json", "rb") as f:
    # with open(f"{base_dir}/vocab_new_0.json", "rb") as f: #just for eval
    with open(f"{base_dir}/vocab_train.json", "rb") as f:
        token_map = json.load(f)

    reverse_answer_lookup = {v: k for k, v in token_map["answer_vocab"].items()}

    with open(f"{base_dir}/train.json", "rb") as f:
        questions_data = json.load(f)
    # print(questions_data)
    #  pdb.set_trace()

    #  with open("/viscam/u/xyang23/clevrer_v2/ceg.json", "rb") as f:
    with open('./clevrer_monet_latents/train-orig-small.json', "rb") as f:
        questions_data = json.load(f)
    questions_data = {int(k): v for k, v in questions_data.items()}
    print('training data loaded')
    with open('./clevrer_monet_latents/valid-orig-small.json', "rb") as f:
        questions_data_test = json.load(f)
    questions_data_test = {int(k) + 10000: v for k, v in questions_data_test.items()}
    print('eval data loaded')

    tf.reset_default_graph()
    model = modellib.ClevrerTransformerModel(**modellib.PRETRAINED_MODEL_CONFIG)
    print('model init')

    inputs_mc = {
        "monet_latents": tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FRAMES, NUM_OBJECTS, modellib.EMBED_DIM], name='monet_latents'),
        "question": tf.placeholder(tf.int32, [BATCH_SIZE, modellib.MAX_QUESTION_LENGTH], name='question'),
        "choices": tf.placeholder(tf.int32, [BATCH_SIZE, modellib.NUM_CHOICES, modellib.MAX_CHOICE_LENGTH], name='choices'),
    }

    output_mc = model.apply_model_mc(inputs_mc)

    # Restore from checkpoint
    # saver = tf.train.Saver()
    # checkpoint_dir = f"{base_dir}/checkpoints/"
    # sess = tf.train.SingularMonitoredSession(checkpoint_dir=checkpoint_dir)
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # saver.restore(sess, ckpt.model_checkpoint_path)

    optimizer = tf.train.AdamOptimizer(0.01)
    labels = tf.placeholder('bool', shape=[BATCH_SIZE, modellib.NUM_CHOICES])
    ground_truth = tf.identity(labels)
    # loss = tf.square(tf.subtract(tf.cast(ground_truth, tf.float32), tf.cast(output_mc, tf.float32)))
    # loss = tf.reduce_mean(tf.square(tf.subtract(tf.cast(ground_truth, tf.float32), tf.sigmoid(output_mc))))
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = bce(tf.cast(ground_truth, tf.int64), output_mc)
    train_op = optimizer.minimize(loss)

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    def train(batch_indices, training_data):
        batch = [[], [], [], []]
        for index in batch_indices:
            monet_latents, question_json, label = training_data[index]
            stride, rem = divmod(monet_latents.shape[0], NUM_FRAMES)
            monet_latents = monet_latents[None, :-rem:stride]
            assert monet_latents.shape[1] == NUM_FRAMES
            question = encode_sentence(
                token_map, question_json["question"], modellib.MAX_QUESTION_LENGTH)
            choices = encode_choices(
                token_map, question_json["choices"])

            for i, data in enumerate([monet_latents, np.expand_dims(question, axis=0), np.expand_dims(choices, axis=0), label]):
                batch[i].append(data)

        lossv, _ = sess.run([loss, train_op], feed_dict={
            inputs_mc["monet_latents"]: np.concatenate(batch[0]),
            inputs_mc["question"]: np.concatenate(batch[1]),
            inputs_mc["choices"]: np.concatenate(batch[2]),
            labels: np.concatenate(batch[3])
        })
        return lossv

    def eval_mc(monet_latents, question_json):
        stride, rem = divmod(monet_latents.shape[0], NUM_FRAMES)
        monet_latents = monet_latents[None, :-rem:stride]
        assert monet_latents.shape[1] == NUM_FRAMES
        question = encode_sentence(token_map, question_json["question"], modellib.MAX_QUESTION_LENGTH)
        choices = encode_choices(token_map, question_json["choices"])
        mc_answer = sess.run(output_mc, feed_dict={
            inputs_mc["monet_latents"]: np.repeat(monet_latents, BATCH_SIZE, axis=0),
            inputs_mc["question"]: np.repeat(np.expand_dims(question, axis=0), BATCH_SIZE, axis=0),
            inputs_mc["choices"]: np.repeat(np.expand_dims(choices, axis=0), BATCH_SIZE, axis=0),
        })
        return mc_answer >= 0

    def evaluate(epoch, eval_data, key='valid'):
        correct_question_cnt = 0
        total_question_cnt = 0
        correct_choice_cnt = 0
        total_choice_cnt = 0
        for sample_scene_idx in jacinle.tqdm(list(eval_data.keys()), desc='Evaluting'):
            for question_json in eval_data[sample_scene_idx]["questions"]:
                true_answer = [choice_json["answer"] == 'correct' for choice_json in question_json["choices"]]
                model_answer = eval_mc(load_monet_latents(base_dir, sample_scene_idx, key), question_json)
                total_question_cnt += 1
                is_question_correct = True

                for answer_idx in range(len(true_answer)):
                    if true_answer[answer_idx] == model_answer[0][answer_idx]:
                        correct_choice_cnt += 1
                    else:
                        is_question_correct = False
                    total_choice_cnt += 1
                if is_question_correct:
                    correct_question_cnt += 1


        print('Epoch', epoch, f'({key})', '\n'
             'correct question', correct_question_cnt, 'out of', total_question_cnt, 'question acc:', correct_question_cnt / total_question_cnt, '\n',
             'correct_choice  ', correct_choice_cnt,   'out of', total_choice_cnt,   'choice acc:', correct_choice_cnt / total_choice_cnt)
        return correct_question_cnt / total_question_cnt, correct_choice_cnt / total_choice_cnt

    training_data = list()
    for scene_idx in questions_data.keys():
        latent = load_monet_latents(base_dir, scene_idx, 'train')
        for question_json in questions_data[scene_idx]["questions"]:
            true_answer = [choice_json["answer"] == 'correct' for choice_json in question_json["choices"]]
            true_answer += [False for _ in range(modellib.NUM_CHOICES - len(true_answer))]
            ans = np.expand_dims(np.array(true_answer),axis=0)
            training_data.append((latent, question_json, ans))

    eval_epoch = 1
    best_question_acc = None
    best_choice_acc = None
    for epoch in range(1, 201):
        indices = list(range(len(training_data)))
        random.shuffle(indices)
        nr_batches, rem = divmod(len(indices), BATCH_SIZE)
        if rem > 0:
            indices = indices[:-rem]

        losses = list()
        for batch_index in jacinle.tqdm(nr_batches, desc=f'epoch={epoch}'):
            lossv = train(indices[batch_index*BATCH_SIZE:batch_index*BATCH_SIZE+BATCH_SIZE], training_data)
            lossv = float(lossv)
            jacinle.get_current_tqdm().set_description(f'epoch={epoch} loss={lossv}')
            losses.append(lossv)
        print('loss: {}.'.format(sum(losses) / len(losses)))

        if epoch % eval_epoch == 0:
            evaluate(epoch, questions_data, 'train')
            question_acc, choice_acc = evaluate(epoch, questions_data_test)

            if best_choice_acc is None or best_choice_acc < choice_acc:
                best_question_acc = question_acc
                best_choice_acc = choice_acc
            print('Best question acc: {}, best choice acc: {}'.format(best_question_acc, best_choice_acc))


if __name__ == "__main__":
    app.run(main)

