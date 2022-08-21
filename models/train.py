"""

python train.py --model_type resnet50_lstm or resnet50_bert --output_dir OUTPUT_DIR --graph_path CEG_PATH --device -1 

"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))

from .description-gen.dataloader import CEGGloveDataset
from .models import resnet_lstm, resnet_bert, utils
def evaluate(model, crit, dataset, opt, device):

    """Evaluate with or without teacher-forcing (beam search decoding)"""
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=False)
    valid_losses = []
    valid_corrects = []
    samples = {}
    for data in tqdm(loader):
        # forward the model to get loss
        fc_feats = data["fc_feats"].to(device)
        answer = data["answer"].to(device)
        answer_mask = data["answer_mask"].to(device)
        answer_len = data["answer_len"]
        question = data["question"].to(device)
        question_len = data["question_len"]
        label = data["label"].to(device, dtype=torch.float32)
        ques_idx = data['ques_idx']
      

        with torch.no_grad():
            logits, probs = model(fc_feats, question, question_len, answer, answer_len)
            loss = crit(logits, label)
            valid_losses.append(loss.item())
            preds = (probs > 0.5).long()
            valid_corrects += preds.eq(label).cpu().numpy().tolist()

        if opt["debug"]:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_losses) / float(len(valid_corrects))
    return valid_acc, valid_loss
    
   
def train(model, crit, dataset, optimizer, lr_scheduler, opt, device, rl_crit=None, results_dir=None):
    """ Train the model """
    model.train()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)

    best_valid_loss = float("inf")
    best_epoch = -1
    early_stopping_cnt = 0
    niter = 0
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()
        
        for batch_idx, data in enumerate(loader):
            if opt['device'] != -1:
                torch.cuda.synchronize()
            fc_feats = data["fc_feats"].to(device)
            if opt["model_type"] == "resnet50_lstm":
                answer = data["answer"].to(device)
                question = data["question"].to(device)
            if opt["model_type"] == "resnet50_bert":
                answer = data["answer_str"].to(device)
                question = data["question_str"].to(device)
            
            question_len = data["question_len"]
            label = data["label"].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            logits, probs = model(fc_feats, question, question_len, answer, answer_len)
            # if (logits.shape != label.shape):
            #     continue
            loss = crit(logits, label)

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            
            if opt['device'] != -1:
                torch.cuda.synchronize() 
       

            preds = (probs > 0.5).long()
            train_corrects = preds.eq(label).cpu().numpy().tolist()


            train_acc = sum(train_corrects) / float(len(train_corrects))
            train_loss = train_loss / float(len(train_corrects))
            print("batch {} epoch {} niter {}, train loss = {:.6f}, train acc = {:.6f}\n".format(batch_idx, epoch, niter, train_loss, train_acc))

            niter += 1
            if opt["debug"]:
                break

        if opt["save_checkpoint_every_n_epoch"] > 0 and epoch % opt["save_checkpoint_every_n_epoch"] == 0:
            model_path = 'epoch_%d.pth' % (epoch)
            torch.save(model.state_dict(), os.path.join(results_dir, model_path)) # torch.save always overwrites if there is an existing checkpoint already
            print("model saved to %s" % (os.path.join(results_dir, model_path)))

        
            # Evaluate using the validation set 
            if opt["validate_every_n_epoch"] > 0 and epoch % opt["validate_every_n_epoch"] == 0:
                dataset.set_phase(opt["validation_phase"])
                valid_acc, valid_loss = evaluate(model, crit, dataset, opt, device)
                

                valid_log_str = "epoch {} niter {}, valid loss = {:.6f}, valid acc = {:.6f}\n".format(epoch, niter, valid_loss, valid_acc)
                print(valid_log_str)
                # with open(model_info_path, 'a') as f:
                #     f.write(valid_log_str)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    utils.symlink(model_path, os.path.join(results_dir, "best_valid.pth"))
                else:
                    early_stopping_cnt += 1
                    if opt["max_es_cnt"] > 0 and early_stopping_cnt >= opt["max_es_cnt"]:
                        break

                # reset to train
                model.train()
                dataset.set_phase(opt["train_phase"])

        if opt["debug"]:
            break


def main(args, opt):
    if opt["device"] >= 0:
        device = torch.device("cuda:%d" % opt["device"])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt["device"])
    else:
        device = torch.device("cpu")

    dataset = CEGGloveDataset(graph_path=args.graph_path ,cur_phase=args.train_phase, data_dir=args.data_dir, feats_dir=args.feats_dir, cache_dir=args.cache_dir, dim_word=args.dim_word, ques_max_len=args.ques_max_len, ans_max_len=args.ans_max_len)
    word2idx = dataset.get_word2idx()
    if "resnet50_lstm" in opt["model_type"]:
        model = resnet_lstm.ResNetLSTM(opt["dim_vid"], opt["dim_hidden"], opt["bidirectional"], dataset.get_vocab_size(), opt["dim_word"], opt["rnn_type"], opt["input_dropout_p"], opt["hidden_dropout_p"], word2idx["<sos>"], word2idx["<eos>"], opt["freeze_embeds"])
    if "resnet50_bert" in opt["model_type"]:
        model = resnet_bert.ResNetBERT(opt["dim_vid"], opt["dim_hidden"], opt["bidirectional"], dataset.get_vocab_size(), opt["dim_word"], opt["input_dropout_p"], opt["hidden_dropout_p"], opt['rnn_type'], word2idx["<sos>"], word2idx["<eos>"], opt["freeze_embeds"])

    if opt["use_glove"]:
        model.load_embedding(dataset.vocab_embedding)
    model = model.to(device)
    if args.model_name_or_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, opt["saved_ckpt"])))

    # crit = utils.LanguageModelCriterion()
    crit = nn.BCEWithLogitsLoss(reduction="mean").to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])
    train(model, crit, dataset, optimizer, exp_lr_scheduler, opt, device, results_dir=args.results_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--graph_path',
        type=str,
        default='',
        help='path to main data')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/viscam/data/clevrer',
        help='path to the training QA json')
    parser.add_argument(
        '--feats_dir',
        nargs='*',
        type=str,
        default=['cache/feats/resnet50/'],
        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument(
        '--dim_word',
        type=int,
        default=300,
        help='the encoding size of each token in the vocabulary, and the video.'
    )
    parser.add_argument(
        "--ques_max_len",
        type=int,
        default=32,
        help='max length of a question')
    parser.add_argument(
        "--ans_max_len",
        type=int,
        default=32,
        help='max length of a answer(containing <sos>,<eos>)')
    #parser.add_argument("--train_phase", type=str, default="train")
    parser.add_argument("--train_phase", type=str, default="cegv3_core_train")
    #parser.add_argument("--validation_phase", type=str, default="validation")
    parser.add_argument("--validation_phase", type=str, default="cegv3_core_validation")


    # Model settings
    parser.add_argument(
        "--model_type", choices=["resnet50_lstm", "resnet_bert"], default='resnet50_lstm', help="which model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Path to a trained model")
    parser.add_argument('--saved_ckpt', type=str, default='best_valid.pth',
                        help='path to saved checkpoint to evaluate')
    parser.add_argument(
        '--rnn_type', choices=["lstm", "gru"], default='lstm', help='lstm or gru')
    parser.add_argument(
        '--dim_vid',
        type=int,
        default=2048,
        help='dim of features of video frames')
    parser.add_argument(
        "--bidirectional",
        type=utils.str2bool,
        default=False,
        help="whether to use bidirectional encoder/decoder.")
    parser.add_argument(
        '--dim_hidden',
        type=int,
        default=512,
        help='size of the rnn hidden layer')
    parser.add_argument(
        '--input_dropout_p',
        type=float,
        default=0.2,
        help='strength of dropout on the input layers')
    parser.add_argument(
        '--hidden_dropout_p',
        type=float,
        default=0.5,
        help='strength of dropout in the Language Model RNN')
    parser.add_argument("--use_glove", type=utils.str2bool, default=True, help="Use Glove to initialize the word embeddings")
    parser.add_argument("--freeze_embeds", type=utils.str2bool, default=True, help="Freeze word embeddings during training")

    # Optimization settings
    parser.add_argument(
        '--epochs', type=int, default=100, help='max number of epochs')
    parser.add_argument("--max_es_cnt", type=int, default=0, help="number of epochs to early stop")
    parser.add_argument(
        '--batch_size', type=int, default=128, help='minibatch size')
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=5,  # 5.,
        help='clip gradients at this value')

    parser.add_argument(
        '--learning_rate', type=float, default=1e-5, help='learning rate')

    parser.add_argument(
        '--learning_rate_decay_every',
        type=int,
        default=200,
        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    # parser.add_argument(
    #     '--optim_alpha', type=float, default=0.9, help='alpha for adam')
    # parser.add_argument(
    #     '--optim_beta', type=float, default=0.999, help='beta used for adam')
    # parser.add_argument(
    #     '--optim_epsilon',
    #     type=float,
    #     default=1e-8,
    #     help='epsilon that goes into denominator for smoothing')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4,
        help='weight_decay. strength of weight regularization')
    parser.add_argument(
        '--save_checkpoint_every_n_epoch',
        type=int,
        default=2,
        help='how often to save a model checkpoint (in epoch)?')
    parser.add_argument("--validate_every_n_epoch", type=int, default=1, help="how often to validate (in epoch)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    # General
    parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")
    parser.add_argument("--debug", type=utils.str2bool, default=False, help="whether to use debug mode (break all the loops)")

    args = parser.parse_args()
    opt = vars(args)
    # check_opt(opt)
    args.results_dir = args.output_dir if args.output_dir is not None else os.path.join("cls_results", args.model_type)
    opt_json = os.path.join(args.results_dir, 'opt_info.json')
    if os.path.isdir(args.results_dir):
        shutil.rmtree(args.results_dir)
    os.mkdir(args.results_dir)
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(args, opt)
