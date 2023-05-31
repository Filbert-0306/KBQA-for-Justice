#encoding:utf-8
from text_model import *
import tensorflow as tf
import keras as kr
import os
import numpy as np
import jieba
import re
import heapq
import codecs
import flask
from gevent import pywsgi
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def predict(sentences):

    config = TextConfig()
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)
    model = TextCNN(config)
    save_dir = './checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')

    _,word_to_id=read_vocab(config.vocab_filename)
    input_x= process_file(sentences,word_to_id,max_length=config.seq_length)
    labels = {0:'从属',
              1:'概念',
              2:'特征',
              3:'法律条文',
              4:'量刑标准',
              5:'其它',
              }

    feed_dict = {
        model.input_x: input_x,
        model.keep_prob: 1,
    }
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    y_prob=session.run(model.prob, feed_dict=feed_dict)
    y_prob=y_prob.tolist()
    # print("y_prob:",y_prob)
    print(sentences)

    prob = y_prob[0]
    #prob = y_prob
    # print("prob:", prob)
    a_array = np.array(prob)
    max_prob = a_array.max()
    # label = a_array.argmax()
    # print("argmax:", label)
    # print("label:", labels[label])
    cat=[]
    top2 = list(map(prob.index, heapq.nlargest(1, prob)))
    cat.append(labels[top2[0]])
    name = labels[top2[0]]
    # print("labels[top2[0]]:", labels[top2[0]])
    # print("top2[0]:", top2[0])
    # print("cat:", cat)

    # for prob in y_prob:
    #     top2= list(map(prob.test, heapq.nlargest(1, prob)))
    #     cat.append(labels[top2[0]])
    # tf.reset_default_graph()
    # return  cat

    return {"name": name, "confidence": float(max_prob)}


def sentence_cut(sentences):
    """
    Args:
        sentence: a list of text need to segment
    Returns:
        seglist:  a list of sentence cut by jieba

    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    seglist=[]
    for sentence in sentences:
        words=[]
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                words.extend(jieba.lcut(blk))
        seglist.append(words)
    return  seglist


def process_file(sentences,word_to_id,max_length=600):
    """
    Args:
        sentence: a text need to predict
        word_to_id:get from def read_vocab()
        max_length:allow max length of sentence
    Returns:
        x_pad: sequence data from  preprocessing sentence

    """
    data_id=[]
    seglist=sentence_cut(sentences)
    for i in range(len(seglist)):
        data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)
    return x_pad


def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id

    """
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]

if __name__ == '__main__':
    app = flask.Flask(__name__)

    @app.route("/service/api/word2vec_cnn_intent_recognize",methods=["GET","POST"])
    def bert_intent_recognize():
        data = {"sucess":0}
        result = None
        param = flask.request.get_json()
        print('param: ', param)
        text = param["text"]
        print('text: ', text)
        #print("type(text): ", type(text))

        tf.reset_default_graph()    # 清空defualt graph以及nodes，一编
        with tf.Session(config=config) as sess:
        #with tf.graph.as_default():
            set_session(sess)
            result = predict(text)


        data["data"] = result
        data["sucess"] = 1

        return flask.jsonify(data)

    server = pywsgi.WSGIServer(("0.0.0.0",60063), app)
    server.serve_forever()

    # sentences = "盗窃罪的概念是什么"
    # print(type(sentences))
    # cat = predict(sentences)    # list类型
    # print(cat)

    # sentences = ['盗窃罪的概念是什么']
    # print(type(sentences))
    # cat = predict(sentences)    # list类型
    # print(cat)

     # r = predict(['盗窃罪的概念是什么'])
     # print(r)


    #print(predict("盗窃罪的法律条文"))
    # with codecs.open('./data/xsbh10_template_test.txt','r',encoding='ansi') as f:   # 原来是utf-8
    #     sample=random.sample(f.readlines(),1)
    #     for line in sample:
    #         try:
    #             line=line.rstrip().split('\t')
    #             assert len(line)==2
    #             sentences.append(line[1])
    #             labels.append(line[0])
    #         except:
    #             pass


    # for i,sentence in enumerate(sentences,0):
    #     print ('----------------------the text-------------------------')
    #     print (sentence[:50]+'....')
    #     print('the orginal label:%s'%labels[i])
    #     #print('the predict label:%s'%cat[i])



