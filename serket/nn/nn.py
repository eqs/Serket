# encoding: utf8
#from __future__ import unicode_literals
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf

import ..serket as srk

__all__ = ['NN', 'save_result', 'train']


def save_result(message, loss_save, loss_save_, save_dir, mode):
    if not os.path.exists( save_dir ):
        os.mkdir( save_dir )

    # messageを保存
    np.savetxt( os.path.join( save_dir, "message_{}.txt".format(mode) ), message, fmt="%f" )
    
    # lossを保存，学習モード時は元のモデルのlossも保存
    if mode == "learn":
        np.savetxt( os.path.join( save_dir, "loss.txt" ), loss_save )
    np.savetxt( os.path.join( save_dir, "loss_message.txt" ), loss_save_ )


def train( data, x, y, m_, graph, loss, train_step, index, num_itr1=5000, num_itr2=5000, save_dir="model", batch_size1=None, batch_size2=None, seaquence_size=None, mode="learn" ):
    N = len(data[0])        # データ数
    D_x = len(data[0][0])   # 入力の次元数
    D_y = len(data[1][0])   # 出力の次元数
    
    # loss保存用のlist
    loss_save = []   # 元のモデル用
    loss_save_ = []  # messageモデル用
    
    # 元のモデルの学習
    if mode=="learn":
        with tf.Session(graph=graph[0]) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
                
            for step in range(1, num_itr1+1):
                # バッチ学習
                if batch_size1==None:
                    _, cur_loss = sess.run([train_step[0], loss[0]], feed_dict={x[0]: data[0], y[0]: data[1]})
                    # 50ごとにloss保存
                    if step % 50 == 0:
                        loss_save.append([step,cur_loss])
                        
                # ミニバッチ学習
                else:
                    if seaquence_size==None:
                        # ランダムインデックスを作成
                        sff_idx = np.random.permutation(N)
                        for idx in range(0, N, batch_size1):
                            # ランダムインデックスをバッチサイズに小分け
                            idx_ = sff_idx[idx: idx + batch_size1 if idx + batch_size1 < N else N]
                            # ミニバッチ作成
                            batch_x = data[0][idx_]
                            batch_y = data[1][idx_]
                            _, cur_loss = sess.run([train_step[0], loss[0]], feed_dict={x[0]: batch_x, y[0]: batch_y})
                    else:
                        # シーケンスサイズのステップでランダムインデックスを作成
                        sff_idx_ = np.arange(0, N, seaquence_size)
                        np.random.shuffle(sff_idx_)
                        sff_idx = np.array([np.arange(sff_idx_[i], sff_idx_[i]+seaquence_size) for i in range(len(sff_idx_))]).flatten()
                        for idx in range(0, N, batch_size1):
                            # ランダムインデックスをバッチサイズに小分け
                            idx_ = sff_idx[idx: idx + batch_size1 if idx + batch_size1 < N else N]
                            # ミニバッチ作成
                            batch_x = data[0][idx_]
                            batch_y = data[1][idx_]
                            _, cur_loss = sess.run([train_step[0], loss[0]], feed_dict={x[0]: batch_x, y[0]: batch_y})
                        
                    # epochごとにloss保存
                    loss_save.append([step,cur_loss])
                
            # モデルの保存
            saver.save(sess, os.path.join(save_dir, "model.ckpt"))

    # message学習
    with tf.Session(graph=graph[1]) as sess:
        sess.run(tf.global_variables_initializer())
        # m(TRAINABLE_VARIABLES[0])以外の変数を復元するため引数を指定
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1:])
        
        # モデルの読み込み(m以外のパラメータを復元)
        saver.restore(sess, os.path.join(save_dir, "model.ckpt"))
        for step in range(1, num_itr2+1):
            # バッチ学習
            if batch_size2==None:
                _, cur_loss_ = sess.run([train_step[1], loss[1]], feed_dict={x[1]: data[0], y[1]: data[1]})
                # 50ごとにloss保存
                if step % 50 == 0:
                    loss_save_.append([step,cur_loss_])
            
            # ミニバッチ学習
            else:
                if seaquence_size==None:
                    # ランダムインデックスを作成
                    sff_idx = np.random.permutation(N)
                    for idx in range(0, N, batch_size2):
                        # ランダムインデックスをバッチサイズに小分け
                        idx_ = sff_idx[idx: idx + batch_size2 if idx + batch_size2 < N else N]
                        # ミニバッチ作成
                        batch_x = data[0][idx_]
                        batch_y = data[1][idx_]
                        _, cur_loss_ = sess.run([train_step[1], loss[1]], feed_dict={x[1]: batch_x, y[1]: batch_y, index: idx_})
                else:
                    # シーケンスサイズのステップでランダムインデックスを作成
                    sff_idx = np.arange(0, N, seaquence_size)
                    np.random.shuffle(sff_idx)
                    sff_idx = np.array([np.arange(sff_idx_[i], sff_idx_[i]+seaquence_size) for i in range(len(sff_idx_))]).flatten()
                    for idx in range(0, N, batch_size2):
                        # ランダムインデックスをバッチサイズに小分け
                        idx_ = sff_idx[idx: idx + batch_size2 if idx + batch_size2 < N else N]
                        # ミニバッチ作成
                        batch_x = data[0][idx_]
                        batch_y = data[1][idx_]
                        _, cur_loss_ = sess.run([train_step[1], loss[1]], feed_dict={x[1]: batch_x, y[1]: batch_y, index: idx_})
                        
                # epochごとにloss保存
                loss_save_.append([step,cur_loss_])

        # messageを出力
        message = sess.run([m_])[0]
                
    # 結果を保存
    save_result(message, loss_save, loss_save_, save_dir, mode) 
    
    return message


class NN(srk.Module, metaclass=ABCMeta):
    def __init__( self, itr1=5000, itr2=5000, name="nn", batch_size1=None, batch_size2=None, seaquence_size=None, mode="learn" ):
        super(NN, self).__init__(name, True)
        self.__itr1 = itr1
        self.__itr2 = itr2
        self.__batch_size1 = batch_size1
        self.__batch_size2 = batch_size2
        self.seaquence_size = seaquence_size
        self.__mode = mode
        self.__n = 0

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
    
    def update( self ):
        data = self.get_observations()
        
        data[0] =  np.array( data[0], dtype=np.float32 ) + 0.01
        data[1] =  np.array( data[1], dtype=np.float32 )
        
        x, y, m_, graph, loss, train_step, index = self.build_model(data)
        
        save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
        
        # NN学習
        message = train( data, x, y, m_, graph, loss, train_step, index, self.__itr1, self.__itr2, save_dir, self.__batch_size1, self.__batch_size2, self.seaquence_size, self.__mode )

        self.__n += 1
        
        # メッセージの送信
        self.send_backward_msgs( [message, None] )
