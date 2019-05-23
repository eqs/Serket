#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

from . import nn
import serket as srk
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod

class NN(srk.Module, metaclass=ABCMeta):
    def __init__( self, itr1=5000, itr2=5000, name="nn", batch_size1=None, batch_size2=None, mode="learn" ):
        super(NN, self).__init__(name, True)
        self.__itr1 = itr1
        self.__itr2 = itr2
        self.__batch_size1 = batch_size1
        self.__batch_size2 = batch_size2
        self.__mode = mode

    def build_model( self, data ):
        N = len(data[0])                # データ数
        input_dim = len(data[0][0])     # 入力の次元数
        output_dim = len(data[1][0])    # 出力の次元数
        
        # 元のモデルのグラフ
        graph1 = tf.Graph()
        with graph1.as_default():
            x1 = tf.placeholder("float", shape=[None, input_dim], name="x")     # 入力(受け取る確率)
            y1 = tf.placeholder("float", shape=[None, output_dim], name="y")    # 出力(観測)
            
            loss, train_step = self.model(x1, y1, input_dim, output_dim)        # 内部を構築
            train_step = train_step.minimize(loss)
        
        # message計算用のモデルのグラフ
        graph2 = tf.Graph()
        with graph2.as_default():
            x2 = tf.placeholder("float", shape=[None, input_dim], name="x")     # 入力(受け取る確率)
            y2 = tf.placeholder("float", shape=[None, output_dim], name="y")    # 出力(観測)
            index = tf.placeholder(tf.int32, shape=[None], name="idx")          # ミニバッチ用インデックス
            m = tf.Variable(tf.zeros([N, input_dim]), name="m")                 # message初期値
            m_ = tf.nn.softmax(m)                                               # message変換(確率)
            
            if self.__batch_size2==None:
                x2_ = tf.multiply(x2, m_)                   # ミニバッチではないときindex不要
            else:
                x2_ = tf.multiply(x2, tf.gather(m_,index))  # ミニバッチであるときindex使用
            
            loss_, train_step_ = self.model(x2_, y2, input_dim, output_dim)     # 内部を構築
            train_step_ = train_step_.minimize(loss_, var_list=[m])             # mのみ変数

        # 各グラフの最適化に必要な情報をlistにまとめる
        x = [x1, x2]
        y = [y1, y2]
        graph = [graph1, graph2]
        loss_list = [loss, loss_]
        train_step_list = [train_step, train_step_]
        
        return x, y, m_, graph, loss_list, train_step_list, index
    
    # クラス継承先でmodelを定義しない時にエラーをはく
    @abstractmethod
    def model( self, x, y, input_dim, output_dim ):
        pass
    
    def update(self):
        data = self.get_observations()
        
        for i in range(2):
            data[i] = np.array( data[i], dtype=np.float32 )
        
        x, y, m_, graph, loss, train_step, index = self.build_model(data)
        
        # NN学習
        message = nn.train( data, x, y, m_, graph, loss, train_step, index, self.__itr1, self.__itr2, self.get_name(), self.__batch_size1, self.__batch_size2, self.__mode )

        # メッセージの送信
        self.send_backward_msgs( [message, None] )
