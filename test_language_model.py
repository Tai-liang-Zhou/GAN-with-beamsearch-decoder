# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:37:34 2018

@author: tom
"""

#import jieba
ge_model = Generator(hps_generator, vocab)
sess_ge, saver_ge, train_dir_ge = setup_training_generator(ge_model)
util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
print("finish load train-generator")
generator_graph = tf.Graph()
with generator_graph.as_default():
    util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
    print("finish load train-generator")
#%%
batchers = batcher.get_batches(mode = 'train')
current_batch = batchers[77]
results = ge_model.beam_search_example(sess_ge, current_batch)
beamsearch_outputs = results['beamsearch_outputs']
beamsearch_outputs = results['Greedy_outputs']

#output_ids = [int(t) for t in beamsearch_outputs[0][0:]]
#decoded_words = data.outputids2words(output_ids, vocab, None)
#results = ge_model.sample_generator(sess_ge, current_batch)
current_batch.original_review_inputs
current_batch.original_review_output
for i in range(5):
    predict_list = np.ndarray.tolist(beamsearch_outputs[:, :, i])
    predict_list = predict_list[0]
    predict_seq = [vocab.id2word(idx) for idx in predict_list]
    print(" ".join(predict_seq))
    
#%%
batchers = batcher.get_batches(mode = 'train')
current_batch = batchers[77]
results = ge_model.greedy_example(sess_ge, current_batch)
output_ids = results['Greedy_outputs'][0]
decoded_words = data.outputids2words(output_ids, vocab, None)
current_batch.original_review_inputs
print("decoded_words :", decoded_words)
current_batch.original_review_output
#%%
loss = results['loss']
train_op = results['train_op']
global_step = results['global_step']
b=  results['decoder_outputs']
encoder_outputs = results['encoder_outputs']
encoder_state = results['encoder_state']
d = results['enc_lens']
e = results['dec_lens']
f = results['encoder_outputs_word']
g = results['sentence_level_input']
h = results['dec_in_state']
decoder_initial_state = results['decoder_initial_state']
encoder_state = results['decoder_initial_state']
dec_in_state = results['dec_in_state']
sentence_level_input = results['sentence_level_input']
encoder_outputs_word = results['encoder_outputs_word']
beamsearch_decoder_initial_state = results['beamsearch_decoder_initial_state']
decoder_initial_state = results['decoder_initial_state']
#%%
jieba.load_userdict('dir.txt')
inputs = input("Enter your ask: ")
sentence = jieba.cut(inputs)
sentence = (" ".join(sentence))
sentence = sentence.split( )
enc_input = [vocab.word2id(w) for w in sentence]
enc_lens = np.array([len(enc_input)])
enc_input = np.array([enc_input])


out_sentence = ('').split( )
dec_batch = [vocab.word2id(w) for w in out_sentence]
dec_batch = [2] + dec_batch
dec_batch.append(3)
while len(dec_batch) < 40:
    dec_batch.append(1)
    
dec_batch = np.array([dec_batch]).shape
dec_batch = np.resize(dec_batch,(1,1,40))
#dec_lens = np.resize(dec_lens,(1,1,40))
dec_lens = np.array([len(dec_batch)])

result = ge_model.run_test_language_model(sess_ge, enc_input, enc_lens, dec_batch , dec_lens)


output_ids = [int(t) for t in result['generated'][0][0]][1:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)
try:
    if decoded_words[0] == '[STOPDOC]':
        decoded_words = decoded_words[1:]
    fst_stop_idx = decoded_words.index(data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
    decoded_words = decoded_words[:fst_stop_idx]
except ValueError:
    decoded_words = decoded_words

if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
    decoded_words.append('.')
decoded_words_all = []
decoded_output = ' '.join(decoded_words).strip()  # single string
decoded_words_all.append(decoded_output)
decoded_words_all = ' '.join(decoded_words_all).strip()
decoded_words_all = decoded_words_all.replace("[UNK] ", "")
decoded_words_all = decoded_words_all.replace("[UNK]", "")
decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
decoded_words_all,_ = re.subn(r"(\. ){2,}", "", decoded_words_all)
if decoded_words_all.startswith('，'):
    decoded_words_all = decoded_words_all[1:]


print("the resonse :",decoded_words_all)
#------------------------------------------------------------------------------------------------


batches = batcher.get_batches(mode = 'test')
batch = batches[2]
batch.original_review_inputs
batch.original_review_output
result = ge_model.max_generator(sess_ge, batch)

output_ids = [int(t) for t in result['generated'][0][0]][0:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)
try:
    if decoded_words[0] == '[STOPDOC]':
        decoded_words = decoded_words[1:]
        fst_stop_idx = decoded_words.index(data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
except ValueError:
    decoded_words = decoded_words
        
if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
    decoded_words.append('.')
decoded_words_all = []
decoded_output = ' '.join(decoded_words).strip()  # single string
decoded_words_all.append(decoded_output)
decoded_words_all = ' '.join(decoded_words_all).strip()
decoded_words_all = decoded_words_all.replace("[UNK] ", "")
decoded_words_all = decoded_words_all.replace("[UNK]", "")
decoded_words_all = decoded_words_all.replace(" ", "")
decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
decoded_words_all,_ = re.subn(r"(\. ){2,}", "", decoded_words_all)
if decoded_words_all.startswith('，'):
    decoded_words_all = decoded_words_all[1:]
print("The resonse   : {}".format(decoded_words_all))







inputs = "好 的 ， 請問 您 需要 什麼 幫忙 ？ 四 。"
sentence = jieba.cut(inputs)
sentence = (" ".join(sentence))
print(sentence)
sentence = sentence.split( )
enc_input = [vocab.word2id(w) for w in sentence]
enc_lens = np.array([len(enc_input)],dtype='int32')
enc_input = np.array([enc_input],dtype='int32')
out_sentence = ('[START]').split( )
dec_batch = [vocab.word2id(w) for w in out_sentence]
while len(dec_batch) < 40:
    dec_batch.append(1)
                
dec_batch = np.array([dec_batch],dtype='int32')
dec_batch = np.resize(dec_batch,(1,1,40))
dec_lens = np.array([len(dec_batch)],dtype='int32')
        
result = ge_model.run_test_language_model(sess_ge, enc_input, enc_lens, dec_batch , dec_lens)
            
output_ids = [int(t) for t in result['generated'][0][0]][0:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)






if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
    decoded_words.append('.')
decoded_words_all = []
decoded_output = ' '.join(decoded_words).strip()  # single string
decoded_words_all.append(decoded_output)
decoded_words_all = ' '.join(decoded_words_all).strip()
decoded_words_all = decoded_words_all.replace("[UNK] ", "")
decoded_words_all = decoded_words_all.replace("[UNK]", "")
decoded_words_all = decoded_words_all.replace(" ", "")
decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
decoded_words_all,_ = re.subn(r"(\. ){2,}", "", decoded_words_all)
if decoded_words_all.startswith('，'):
    decoded_words_all = decoded_words_all[1:]
print("The resonse   : {}".format(decoded_words_all))

