# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
#import tensorflow.contrib.keras as kr
import keras as kr
import json
import flask
import pickle
import numpy as np
from gevent import pywsgi
from keras.backend.tensorflow_backend import set_session

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

global graph, model, sess

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
graph = tf.get_default_graph()
set_session(sess)

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/justice'
vocab_dir = os.path.join(base_dir, 'xsbh10_template_vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'justice_best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, text):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(text)
        # print("cat_to_id:", self.cat_to_id)
        # print("word_to_id:", self.word_to_id)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        # print("data:", data)
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        # print("feed_dict:", feed_dict)
        '''
        作用：softmax函数的作用就是归一化。
        输入: 全连接层（往往是模型的最后一层）的值，一般代码中叫做logits
        输出: 归一化的值，含义是属于该位置的概率，一般代码叫做probs。例如输入[0.4,0.1,0.2,0.3],那么这个样本最可能属于第0个位置，也就是第0类。这是由于logits的维度大小就设定的是任务的类别，所以第0个位置就代表第0类。softmax函数的输出不改变维度的大小。
        用途：如果做单分类问题，那么输出的值就取top1(最大，argmax)；如果做多(N)分类问题，那么输出的值就取topN
        '''
        probs = self.session.run(self.model.logits, feed_dict=feed_dict)
        probs_list = sess.run(tf.nn.softmax(probs[0]))
        probs_list.sort()   # softmax后，排序取最大值
        max_prob = probs_list[len(probs_list)-1]

        # print("probs: ", probs)
        # print("probs[0]: ", probs[0])
        # print("probs_type: ", type(probs))
        # print("softmax: ", tf.argmax(tf.nn.softmax(self.model.logits), 1))
        # print("softmax_prob: ", sess.run(tf.nn.softmax(probs)))
        # print("softmax_prob[0]: ", sess.run(tf.nn.softmax(probs[0])))
        # print("max_prob: ", max_prob)
        # print("probs_list: ", probs_list)

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        # print("y_pred_cls:", y_pred_cls)
        name = self.categories[y_pred_cls[0]]
        return {"name": name, "confidence": float(max_prob)}


cnn_model = CnnModel()

if __name__ == '__main__':
    app = flask.Flask(__name__)

    @app.route("/service/api/cnn_intent_recognize",methods=["GET","POST"])
    def bert_intent_recognize():
        data = {"sucess":0}
        result = None
        param = flask.request.get_json()
        print('param: ', param)
        text = param["text"]
        print('text: ', text)
        print("type(text): ", type(text))
        with graph.as_default():
            set_session(sess)
            result = cnn_model.predict(text)

        data["data"] = result
        data["sucess"] = 1

        return flask.jsonify(data)

    server = pywsgi.WSGIServer(("0.0.0.0",60062), app)
    server.serve_forever()

    '''cnn_model = CnnModel()
        test_demo = ['抢劫的上级罪名是什么',
                     '强奸怎么判?']
        for i in test_demo:
            print(cnn_model.predict(i))'''

     # r = cnn_model.predict("抢劫罪的概念是什么")
     # print(r)