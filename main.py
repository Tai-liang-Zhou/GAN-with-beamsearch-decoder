"""
Created on Wed May 23 13:33:39 2018
tensorflow : 1.6
cuda :9.0
@author: tom
"""
#import jieba
import codecs
import json
import math
import os
import re
import time
from collections import namedtuple

import nltk
import numpy as np
import opencc
import tensorflow as tf
from tqdm import tqdm

import batch_discriminator as bd
import data
import util
from batch_discriminator import DisBatcher
from batcher import Batch, Example, GenBatcher
from data import Vocab
from discriminator import Discriminator
from generated_sample import Generated_sample
from model import Generator
from pyltp import Segmentor

FLAGS = tf.flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0
segmentor = Segmentor()
segmentor.load("./ltp_data_v3.4.0/cws.model")
t2s = opencc.OpenCC('t2s')
s2t = opencc.OpenCC('s2t')

# Where to find data 去哪裡找data
tf.flags.DEFINE_string(
    'data_path', 'review_generation_dataset/train/* ',
    'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.flags.DEFINE_string('vocab_path', 'new_data/new_vocab.txt',
                       'Path expression to text vocabulary file.')

# Important settings 匯入設定
tf.flags.DEFINE_string(
    'mode', 'test_language_model',
    'must be one of adversarial_train/train_generator/train_discriminator/test_language_model')

# Where to save output 儲存檔案
tf.flags.DEFINE_string('log_root', './', 'Root directory for all logging.')
tf.flags.DEFINE_string(
    'exp_name', 'myexperiment',
    'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

tf.flags.DEFINE_integer('gpuid', 1, 'for gradient clipping')
tf.flags.DEFINE_string('dataset', 'yelp', "dataset which you use")
tf.flags.DEFINE_string('run_method', 'auto-encoder', 'must be one of auto-encoder/language_model')

tf.flags.DEFINE_integer('max_enc_sen_num', 1,
                        'max timesteps of encoder (max source text tokens)')  # for discriminator
tf.flags.DEFINE_integer('max_enc_seq_len', 40,
                        'max timesteps of encoder (max source text tokens)')  # for discriminator

tf.flags.DEFINE_integer('max_dec_sen_num', 1,
                        'max timesteps of decoder (max source text tokens)')  # for generator
tf.flags.DEFINE_integer('max_dec_steps', 40,
                        'max timesteps of decoder (max source text tokens)')  # for generator
# Hyperparameters
# for discriminator and generator
tf.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
# for discriminator and generator
tf.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
# for discriminator and generator
tf.flags.DEFINE_integer('batch_size', 1, 'minibatch size')
tf.flags.DEFINE_integer('max_enc_steps', 50,
                        'max timesteps of encoder (max source text tokens)')  # for generator
tf.flags.DEFINE_integer(
    'min_dec_steps', 35,
    'Minimum sequence length of generated summary. Applies only for beam search decoding mode'
)  # for generator
# tf.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.flags.DEFINE_integer(
    'vocab_size', 66317,
    'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.'
)
# for discriminator and generator
tf.flags.DEFINE_float('lr', 0.1, 'learning rate')
# for discriminator and generator
tf.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
# for discriminator and generator
tf.flags.DEFINE_float('rand_unif_init_mag', 0.02,
                      'magnitude for lstm cells random uniform inititalization')
# for discriminator and generator
tf.flags.DEFINE_float('trunc_norm_init_std', 1e-4,
                      'std of trunc norm init, used for initializing everything else')
# for discriminator and generator
tf.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
# beamsearch decoder
tf.flags.DEFINE_string('beamsearch', 'beamsearch_test',
                       'must be one of beamsearch_train/beamsearch_test/')
tf.flags.DEFINE_integer('beam_width', 5, 'beam search width')


def setup_training_generator(model):
    train_dir = os.path.join(FLAGS.log_root, "train-generator")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    sess = tf.Session(config=util.get_config())
    model.build_graph(sess)  # build the graph
    # we use this to load checkpoints for decoding
    saver = tf.train.Saver(max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess, saver, train_dir


def setup_training_discriminator(model):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train-discriminator")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    model.build_graph()  # build the graph
    # we use this to load checkpoints for decoding
    saver = tf.train.Saver(max_to_keep=5)
    sess = tf.Session(config=util.get_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess, saver, train_dir


def batch_to_batch(batch, batcher, dis_batcher):

    db_example_list = []

    for i in range(FLAGS.batch_size):
        new_dis_example = bd.Example(batch.original_review_inputs[i], -0.01, dis_batcher._vocab,
                                     dis_batcher._hps)
        db_example_list.append(new_dis_example)
    return bd.Batch(db_example_list, dis_batcher._hps, dis_batcher._vocab)


def output_to_batch(current_batch, results, batcher, dis_batcher):
    # 生成新的 batch 和 dis-batch
    example_list = []
    db_example_list = []

    for i in range(FLAGS.batch_size):
        decoded_words_all = []
        encode_words = current_batch.original_review_inputs[i]

        for j in range(FLAGS.max_dec_sen_num):
            output_ids = [int(t) for t in results['Greedy_outputs'][i][0:]]
            decoded_words = data.outputids2words(output_ids, batcher._vocab, None)
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                # index of the (first) [STOP] symbol
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            if len(decoded_words) < 2:
                continue
            if len(decoded_words_all) > 0:
                new_set1 = set(decoded_words_all[len(decoded_words_all) - 1].split())
                new_set2 = set(decoded_words)
                if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                    continue
            if decoded_words[-1] != '.' and decoded_words[-1] != '!' and decoded_words[-1] != '?':
                decoded_words.append('.')
            decoded_output = ' '.join(decoded_words).strip()
            decoded_words_all.append(decoded_output)

        decoded_words_all = ' '.join(decoded_words_all).strip()
        try:
            fst_stop_idx = decoded_words_all.index(data.STOP_DECODING_DOCUMENT)
            decoded_words_all = decoded_words_all[:fst_stop_idx]
        except ValueError:
            decoded_words_all = decoded_words_all
        decoded_words_all = decoded_words_all.replace("[UNK] ", "")
        decoded_words_all = decoded_words_all.replace("[UNK]", "")
        decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
        decoded_words_all, _ = re.subn(r"(\. ){2,}", "", decoded_words_all)

        if decoded_words_all.strip() == "":
            new_dis_example = bd.Example(current_batch.original_review_output[i], -0.0001,
                                         dis_batcher._vocab, dis_batcher._hps)
            new_example = Example(current_batch.original_review_output[i], batcher._vocab,
                                  batcher._hps, encode_words)
        else:
            new_dis_example = bd.Example(decoded_words_all, 1, dis_batcher._vocab, dis_batcher._hps)
            new_example = Example(decoded_words_all, batcher._vocab, batcher._hps, encode_words)
        example_list.append(new_example)
        db_example_list.append(new_dis_example)
    return Batch(example_list, batcher._hps, batcher._vocab), bd.Batch(
        db_example_list, dis_batcher._hps, dis_batcher._vocab)


def run_train_generator(model, discirminator_model, discriminator_sess, batcher, dis_batcher,
                        batches, sess, saver, train_dir):
    tf.logging.info("starting training generator")
    step = 0
    t0 = time.time()
    loss_window = 0.0
    new_loss_window = 0.0

    while step < len(batches):
        current_batch = batches[step]
        step += 1
        for i in range(1):
            results = model.greedy_example(sess, current_batch)
            new_batch, new_dis_batch = output_to_batch(current_batch, results, batcher, dis_batcher)
            # output_to_batch new-batch dis-batch

            reward = discirminator_model.run_ypred_auc(discriminator_sess, new_dis_batch)
            reward_sentence_level = reward['y_pred_auc_sentence']
            for i in range(len(reward['y_pred_auc'])):
                for j in range(len(reward['y_pred_auc'][i])):
                    for k in range(len(reward['y_pred_auc'][i][j])):
                        reward['y_pred_auc'][i][j][k] = reward['y_pred_auc'][i][j][k] + 1e10

            reward['y_pred_auc'] = np.reshape(
                np.array(reward['y_pred_auc']), [
                    batcher._hps.batch_size * batcher._hps.max_dec_sen_num,
                    batcher._hps.max_dec_steps
                ])

            # 關鍵步驟reward
            results = model.run_train_step(sess, new_batch, reward['y_pred_auc'])

            loss = results['loss']
            loss_window += loss
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

        new_dis_batch = batch_to_batch(current_batch, batcher, dis_batcher)

        reward = discirminator_model.run_ypred_auc(discriminator_sess, new_dis_batch)
        reward_sentence_level = reward['y_pred_auc_sentence']

        for i in range(len(reward['y_pred_auc'])):  # i = batch_size
            # j = how many sentence is.
            for j in range(len(reward['y_pred_auc'][i])):
                # k = how many word is.
                for k in range(len(reward['y_pred_auc'][i][j])):
                    reward['y_pred_auc'][i][j][k] = reward['y_pred_auc'][i][j][k] + 1e10

        reward['y_pred_auc'] = np.reshape(
            np.array(reward['y_pred_auc']),
            [FLAGS.batch_size * batcher._hps.max_dec_sen_num, batcher._hps.max_dec_steps])
        new_results = model.run_train_step(sess, current_batch,
                                           reward['y_pred_auc'])  # 用新的reward去算很重要

        new_loss = new_results['loss']
        new_loss_window += new_loss
        if not np.isfinite(new_loss):
            raise Exception("new Loss is not finite. Stopping.")
        # we need this to update our running average loss
        train_step = new_results['global_step']
        tf.logging.info('the global step of generator : %d', train_step)
        if train_step % 100 == 0:
            saver.save(sess, train_dir + "/model", global_step=train_step)
    t1 = time.time()
    tf.logging.info('seconds for %d training generator step: %.3f ', train_step,
                    (t1 - t0) / len(batches))
    tf.logging.info('loss: %f',
                    loss_window / (len(batches) / len(batches)))  # print the loss to screen
    tf.logging.info('teach forcing loss: %f',
                    new_loss_window / len(batches))  # print the loss to screen


def run_pre_train_generator(model, batcher, max_run_epoch, sess, saver, train_dir):
    tf.logging.info("starting run_pre_train_generator")
    epoch = 0
    # while epoch < max_run_epoch:
    for epoch in tqdm(range(max_run_epoch)):
        batchers = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        # print(len(batchers))
        while step < len(batchers):
            current_batch = batchers[step]
            results = model.run_pre_train_step(sess, current_batch)

            loss = results['loss']
            loss_window += loss
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")
            perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
            tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" %
                       (results['global_step'], results['loss'], perplexity))
            # print('epoch: {}, loss : {:.3f} , generator global step : {}'.format(
            #     epoch, results['loss'], results['global_step']))
            # we need this to update our running average loss
            train_step = results['global_step']
            if train_step % 100 == 0:

                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step,
                                (t1 - t0) / 100)
                t0 = time.time()
                # print the loss to screen
                tf.logging.info('loss: %f', loss_window / 100)
                loss_window = 0.0
                saver.save(sess, train_dir + "/model", global_step=train_step)

            step += 1
        epoch += 1
        tf.logging.info("finished %d epoches", epoch)


def run_pre_train_discriminator(model_dis, batcher, max_run_epoch, sess, saver, train_dir):
    tf.logging.info("starting run_pre_train_discriminator")

    epoch = 0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            results = model_dis.run_pre_train_step(sess, current_batch)
            # print('discriminator global_step : {}, out_loss_sentence : {}'.format(results['global_step'], results['out_loss_sentence']))
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            # we need this to update our running average loss
            train_step = results['global_step']
            if train_step % 100 == 0:
                print('discriminator global_step : {}, epoch : {}'.format(
                    results['global_step'], epoch))
                t1 = time.time()
                tf.logging.info('seconds for %d training dirscriminator step: %.3f ', train_step,
                                (t1 - t0) / 100)
                t0 = time.time()
                # print the loss to screen
                tf.logging.info('loss: %f', loss_window / 100)
                loss_window = 0.0

            if train_step % 100 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                run_test_discriminator(model_dis, batcher, sess, str(train_step))

        epoch += 1
        tf.logging.info("finished %d epoches", epoch)


def run_test_discriminator(model, batcher, sess, train_step):
    tf.logging.info("starting run testing discriminator")

    discriminator_file = codecs.open("discriminator_result/" + train_step + "discriminator.txt",
                                     "w", "utf-8")

    batches = batcher.get_batches(mode="test")
    step = 0
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        result = model.run_ypred_auc(sess, current_batch)
        outloss = result['y_pred_auc']
        outloss_sentence = result['y_pred_auc_sentence']

        for i in range(FLAGS.batch_size):
            for j in range(batcher._hps.max_enc_sen_num):
                a = {
                    "example": current_batch.review_sentenc_orig[i][j],
                    "score": [np.float64(outloss[i][j][k]) for k in range(len(outloss[i][j]))],
                    "sentence_level_score": np.float64(outloss_sentence[i][j])
                }
                string_a = json.dumps(a, ensure_ascii=False)
                discriminator_file.write(string_a + "\n")
    discriminator_file.close()
    return 0


def run_train_discriminator(model,
                            max_epoch,
                            batcher,
                            batches,
                            sess,
                            saver,
                            train_dir,
                            whole_decay=False):
    tf.logging.info("starting trining discriminator")
    step = 0
    t0 = time.time()
    loss_window = 0.0
    epoch = 0
    while epoch < max_epoch:
        epoch += 1

        while step < len(batches):

            current_batch = batches[step]
            step += 1
            results = model.run_pre_train_step(sess, current_batch)

            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            # we need this to update our running average loss
            train_step = results['global_step']
            tf.logging.info('the globa steps of discriminator: %d', train_step)
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training dirscriminator step: %.3f ', train_step,
                                (t1 - t0) / 100)
                t0 = time.time()
                # print the loss to screen
                tf.logging.info('loss: %f', loss_window / 100)
                loss_window = 0.0
            if train_step % 100 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                run_test_discriminator(model, batcher, sess, str(train_step))
    return whole_decay


def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


# %%


def main(unused_argv):
    # %%
    # choose what level of logging you want
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting running in %s mode...', (FLAGS.mode))
    # 創建字典
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    hparam_list = [
        'mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std',
        'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_sen_num', 'max_dec_steps',
        'max_enc_steps'
    ]
    hps_dict = {}
    for key, val in FLAGS.__flags.items():
        if key in hparam_list:
            hps_dict[key] = val.value  # add it to the dict
    hps_generator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    hparam_list = [
        'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
        'hidden_dim', 'emb_dim', 'batch_size', 'max_enc_sen_num', 'max_enc_seq_len'
    ]
    hps_dict = {}

    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:
            hps_dict[key] = val.value  # add it to the dict
    hps_discriminator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # # 取出最小batch size 的資料量
    batcher = GenBatcher(vocab, hps_generator)
    # print(batcher.train_batch[0].original_review_inputs)
    # print(len(batcher.train_batch[0].original_review_inputs))
    tf.set_random_seed(123)
    # %%
    if FLAGS.mode == 'train_generator':

        # print("Start pre-training ......")
        ge_model = Generator(hps_generator, vocab)
        sess_ge, saver_ge, train_dir_ge = setup_training_generator(ge_model)

        generated = Generated_sample(ge_model, vocab, batcher, sess_ge)
        print("Start pre-training generator......")
        run_pre_train_generator(ge_model, batcher, 300, sess_ge, saver_ge, train_dir_ge)
        # util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
        print("finish load train-generator")

        print("Generating negative examples......")
        generator_graph = tf.Graph()
        with generator_graph.as_default():
            util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
            print("finish load train-generator")

        generated.generator_train_negative_example()
        generated.generator_test_negative_example()

        print("finish write")
    elif FLAGS.mode == 'train_discriminator':
        # print("Start pre-training ......")
        model_dis = Discriminator(hps_discriminator, vocab)
        dis_batcher = DisBatcher(hps_discriminator, vocab, "discriminator_train/positive/*",
                                 "discriminator_train/negative/*", "discriminator_test/positive/*",
                                 "discriminator_test/negative/*")
        sess_dis, saver_dis, train_dir_dis = setup_training_discriminator(model_dis)

        print("Start pre-training discriminator......")
        if not os.path.exists("discriminator_result"):
            os.mkdir("discriminator_result")
        run_pre_train_discriminator(model_dis, dis_batcher, 1000, sess_dis, saver_dis,
                                    train_dir_dis)

    elif FLAGS.mode == "adversarial_train":

        generator_graph = tf.Graph()
        discriminatorr_graph = tf.Graph()

        print("Start adversarial-training......")
        # tf.reset_default_graph()

        with generator_graph.as_default():
            model = Generator(hps_generator, vocab)
            sess_ge, saver_ge, train_dir_ge = setup_training_generator(model)
            generated = Generated_sample(model, vocab, batcher, sess_ge)

            util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
            print("finish load train-generator")
        with discriminatorr_graph.as_default():
            model_dis = Discriminator(hps_discriminator, vocab)
            dis_batcher = DisBatcher(hps_discriminator, vocab, "discriminator_train/positive/*",
                                     "discriminator_train/negative/*",
                                     "discriminator_test/positive/*",
                                     "discriminator_test/negative/*")
            sess_dis, saver_dis, train_dir_dis = setup_training_discriminator(model_dis)

            util.load_ckpt(saver_dis, sess_dis, ckpt_dir="train-discriminator")
            print("finish load train-discriminator")

        print("Start adversarial  training......")
        if not os.path.exists("train_sample_generated"):
            os.mkdir("train_sample_generated")
        if not os.path.exists("test_max_generated"):
            os.mkdir("test_max_generated")
        if not os.path.exists("test_sample_generated"):
            os.mkdir("test_sample_generated")

        whole_decay = False

        for epoch in range(100):
            print('開始訓練')
            batches = batcher.get_batches(mode='train')
            for step in range(int(len(batches) / 14)):

                run_train_generator(model, model_dis, sess_dis, batcher, dis_batcher,
                                    batches[step * 14:(step + 1) * 14], sess_ge, saver_ge,
                                    train_dir_ge)
                generated.generator_sample_example(
                    "train_sample_generated/" + str(epoch) + "epoch_step" + str(step) +
                    "_temp_positive", "train_sample_generated/" + str(epoch) + "epoch_step" +
                    str(step) + "_temp_negative", 14)

                tf.logging.info("test performance: ")
                tf.logging.info("epoch: " + str(epoch) + " step: " + str(step))

                #                print("evaluate the diversity of DP-GAN (decode based on  max probability)")
                #                generated.generator_test_sample_example(
                #                    "test_sample_generated/" +
                #                    str(epoch) + "epoch_step" + str(step) + "_temp_positive",
                #                    "test_sample_generated/" + str(epoch) + "epoch_step" + str(step) + "_temp_negative", 14)
                #
                #                print("evaluate the diversity of DP-GAN (decode based on sampling)")
                #                generated.generator_test_max_example(
                #                    "test_max_generated/" +
                #                    str(epoch) + "epoch_step" + str(step) + "_temp_positive",
                #                    "test_max_generated/" + str(epoch) + "epoch_step" + str(step) + "_temp_negative", 14)

                dis_batcher.train_queue = []
                for i in range(epoch + 1):
                    for j in range(step + 1):
                        dis_batcher.train_queue += dis_batcher.fill_example_queue(
                            "train_sample_generated/" + str(i) + "epoch_step" + str(j) +
                            "_temp_positive/*")
                        dis_batcher.train_queue += dis_batcher.fill_example_queue(
                            "train_sample_generated/" + str(i) + "epoch_step" + str(j) +
                            "_temp_negative/*")
                dis_batcher.train_batch = dis_batcher.create_batches(mode="train", shuffleis=True)
                whole_decay = run_train_discriminator(model_dis, 5, dis_batcher,
                                                      dis_batcher.get_batches(mode="train"),
                                                      sess_dis, saver_dis, train_dir_dis,
                                                      whole_decay)
    elif FLAGS.mode == "test_language_model":
        ge_model = Generator(hps_generator, vocab)
        sess_ge, saver_ge, train_dir_ge = setup_training_generator(ge_model)
        util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
        print("finish load train-generator")

        #        generator_graph = tf.Graph()
        #        with generator_graph.as_default():
        #            util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
        #            print("finish load train-generator")

        # jieba.load_userdict('dir.txt')
        inputs = ''
        while inputs != "close":

            inputs = input("Enter your ask: ")
            sentence = segmentor.segment(t2s.convert(inputs))
            #            sentence = jieba.cut(inputs)
            sentence = (" ".join(sentence))
            sentence = s2t.convert(sentence)
            print(sentence)
            sentence = sentence.split()

            enc_input = [vocab.word2id(w) for w in sentence]
            enc_lens = np.array([len(enc_input)])
            enc_input = np.array([enc_input])
            out_sentence = ('[START]').split()
            dec_batch = [vocab.word2id(w) for w in out_sentence]
            #dec_batch = [2] + dec_batch
            # dec_batch.append(3)
            while len(dec_batch) < 40:
                dec_batch.append(1)

            dec_batch = np.array([dec_batch])
            dec_batch = np.resize(dec_batch, (1, 1, 40))
            dec_lens = np.array([len(dec_batch)])
            if (FLAGS.beamsearch == 'beamsearch_train'):
                result = ge_model.run_test_language_model(sess_ge, enc_input, enc_lens, dec_batch,
                                                          dec_lens)
                #                print(result['generated'])
                #                print(result['generated'].shape)
                output_ids = result['generated'][0]
                decoded_words = data.outputids2words(output_ids, vocab, None)
                print("decoded_words :", decoded_words)
            else:
                results = ge_model.run_test_beamsearch_example(sess_ge, enc_input, enc_lens,
                                                               dec_batch, dec_lens)
                beamsearch_outputs = results['beamsearch_outputs']
                for i in range(5):
                    predict_list = np.ndarray.tolist(beamsearch_outputs[:, :, i])
                    predict_list = predict_list[0]
                    predict_seq = [vocab.id2word(idx) for idx in predict_list]
                    decoded_words = " ".join(predict_seq).split()
                    #                    decoded_words = decoded_words

                    try:
                        if decoded_words[0] == '[STOPDOC]':
                            decoded_words = decoded_words[1:]
                        # index of the (first) [STOP] symbol
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if decoded_words[-1] != '.' and decoded_words[-1] != '!' and decoded_words[
                            -1] != '?':
                        decoded_words.append('.')
                    decoded_words_all = []
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                    decoded_words_all = ' '.join(decoded_words_all).strip()
                    decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                    decoded_words_all = decoded_words_all.replace("[UNK]", "")
                    decoded_words_all = decoded_words_all.replace(" ", "")
                    decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
                    decoded_words_all, _ = re.subn(r"(\. ){2,}", "", decoded_words_all)
                    if decoded_words_all.startswith('，'):
                        decoded_words_all = decoded_words_all[1:]
                    print("The resonse   : {}".format(decoded_words_all))


if __name__ == '__main__':
    nltk.download('punkt')
    tf.app.run()
