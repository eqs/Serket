# -*- coding: utf-8 -*-
#from __future__ import unicode_literals
import os
import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.io import wavfile
from librosa import logamplitude
from librosa.feature import delta, melspectrogram
from librosa.filters import dct

import ..serket as srk

__all__ = ['HACFeatureExtracter', 'NFCC_PARAMS']


MFCC_PARAMS = {
    'n_mfcc': 13,   # Librosa default is 20
    #'n_fft': 320,   # Librosa default is 2048
    #'hop_length': 160,  # Librosa default is 512
    #'n_mels': 30,  # Librosa default is 128
}

# パラメータを置き換える
def complete_mfcc_params(values):
    mfcc_params = MFCC_PARAMS.copy()
    mfcc_params.update(values)
    return mfcc_params

# mfccの計算
def mfcc(data, sr=22050, n_mfcc=20, **kwargs):
    S = logamplitude(melspectrogram(y=data, sr=sr, **kwargs))
    return np.dot(dct(n_mfcc, S.shape[0]), S)

# コードブックの計算
def build_codebooks_from_list_of_wav(wavs, ks, **mfcc_params):
    mfccs = []
    for w in wavs:
        sr, data = wavfile.read(w)
        cur_mfccs = mfcc(data, sr=sr, **complete_mfcc_params(mfcc_params))
        mfccs.append(cur_mfccs)
        cdb_mfcc, _ = kmeans2(np.vstack([m.T for m in mfccs]), ks[0])
        cdb_dmfcc, _ = kmeans2(np.vstack([delta(m).T for m in mfccs]), ks[1])
        cdb_ddmfcc, _ = kmeans2(np.vstack([delta(m, order=2).T for m in mfccs]), ks[2])
        return (cdb_mfcc, cdb_dmfcc, cdb_ddmfcc)

# 数え上げ
def coocurrences(quantized_data, n_quantized, lag):
    pair_idx = quantized_data[lag:] * n_quantized + quantized_data[:-lag]
    return np.bincount(pair_idx, minlength=n_quantized ** 2)

def compute_coocurrences(data, centroids, lags):
    quantized, _ = vq(data, centroids)
    coocs = [coocurrences(quantized, centroids.shape[0], l) for l in lags]
    return np.hstack(coocs)

# hac計算
def hac(data, sr, codebooks, lags=[5, 2], **mfcc_params):
    mfccs = mfcc(data, sr=sr, **complete_mfcc_params(mfcc_params))
    d_mfccs = delta(mfccs)
    dd_mfccs = delta(mfccs, order=2)
    streams = [mfccs.T, d_mfccs.T, dd_mfccs.T]
    return np.hstack([compute_coocurrences(stream, codebook, lags) for (stream, codebook) in zip(streams, codebooks)])

# wavを読み込みhacへ変換
def wav2hac(wav_path, codebooks, lags=[5, 2], **mfcc_params):
    sr, data = wavfile.read(wav_path)
    return hac(data, sr, codebooks, lags=lags, **mfcc_params)    


class HACFeatureExtractor(srk.Module):
    def __init__(self, filenames, ks, lags=[5,2], name="HACFeatureExtracter", **mfcc_params):
        super(HACFeatureExtractor, self).__init__(name, False)
        self.is_ninitilized = False
        self.filenames = filenames
        self.ks = ks
        self.lags = lags
        self.mfcc_params = mfcc_params
        
        # コードブックの作成
        cdbs = build_codebooks_from_list_of_wav( self.filenames, self.ks, **self.mfcc_params )
        
        # wavをhacへ変換
        hacs = []
        for n in range(len(self.filenames)):
            hacs.append( wav2hac(self.filenames[n], cdbs, self.lags, **self.mfcc_params) )
        
        # hacsを保存
        save_dir = self.get_name()
        try:
            os.mkdir( save_dir )
        except:
            pass
        np.savetxt( os.path.join( save_dir, "hac.txt"), hacs )

        self.is_ninitilized = True
    
        self.set_forward_msg( hacs )

