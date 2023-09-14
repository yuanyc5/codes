# _*_coding:utf-8_*_
#keras包
import numpy as np 
import pandas as pd 
import collections
import logging
import sys,os
#from keras import utils as np_utils
from Bio import SeqIO,motifs
from Bio.Seq import Seq
from sklearn.preprocessing import scale,MinMaxScaler
#from sklearn import preprocessing
from sklearn.model_selection import KFold
#from sklearn.preprocessing import LabelEncoder#对标签进行编码
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import scipy.special##############
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
#from gensim.models.doc2vec import Doc2Vec,LabeledSentence#########################
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from keras_utils import Capsule
import tensorflow as tf
#from tfdeterminism import patch
#patch()
graph = tf.get_default_graph()#################
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.optimizers import Adam,SGD,RMSprop,Adagrad
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
from keras.models import load_model,Sequential,Model
from keras.backend import *
from keras.layers import *
import time
from keras.callbacks import TensorBoard,EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,Callback, LearningRateScheduler
from keras.layers import *
from keras.preprocessing import sequence
import os
import keras
from keras_multi_head import MultiHeadAttention
from keras.layers.wrappers import Bidirectional
from Bio import SeqIO,motifs
from Bio.Seq import Seq
global keys
import random
import itertools
from keras.callbacks import Callback
from keras.optimizers import Optimizer
import keras.backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#K.clear_session()#####################

from keras import initializers, regularizers, constraints, optimizers

#from keras.engine.topology import Layer
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from keras.layers import Activation,Layer,Softmax, Concatenate, Convolution1D, Dropout, BatchNormalization, Dense, Flatten, MaxPooling1D
#from tensorflow.keras import Layer
#from keras.engine import Layer
### Avoid warning ###
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import math
from keras.layers import Highway
from keras.layers.wrappers import Bidirectional
#from Capsule import Capsule
#from Capsule_Keras import *
from keras_utils import Capsule
from attention import Position_Embedding,Position_Embedding_word, Attention,Attention_word,Attention_weight,Attention_word_weight
#from keras_multi_head import MultiHead, MultiHeadAttention
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#tf.get_variable('weight',initializer=tf.truncated_normal_initializer(mean=0,stddev=0.02,seed=1))
config = ConfigProto()###########
#config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)##############


tf.compat.v1.Session(config=config)

random.seed(48)#为python设置随机种子
os.environ['PYTHONHASHSEED']=str(48)#哈希随机种子
os.environ['TF_DETERMINISTIC_OPS']='1'#
np.random.seed(48)#为numpy设置随机种子
tf.set_random_seed(48)#设置全局随机种子



#固定随机种子

    
def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)
        
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index   


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index    


def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]        
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index  

def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((len(seq), len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i+kmer]]]
        vectors[i][coden_dict[seq[i:i+kmer]]] = value/len(seq)
    return vectors

def frequency(seq,kmer,coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i+k]
        kmer_value = coden_dict[kmer]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict

def phy90(seq):
    di_prop = pd.read_csv('./DNA_Di_Prop.txt')
    di_prop = di_prop.iloc[:, 1:]
    scaled_di_prop = scale(di_prop, axis=1) # Standardization
    di_cols = di_prop.columns.tolist()
    di_prop = pd.DataFrame(scaled_di_prop, columns=di_cols)
    pp_di = {}
    
    for i in range(16):
        key = di_prop.columns[i]
        items = di_prop.iloc[:, i].tolist()
        pp_di[key] = items
 
    seqLength = len(seq)
    sequence_vector = np.zeros([len(seq), 90])
    k = 2
    for i in range(seqLength - int(k) + 1):
        sequence_vector[i, 0:90] = pp_di[seq[i:i+k]]
    return sequence_vector

def phy12(seq):
    tri_prop = pd.read_csv('./DNA_Tri_Prop.txt')
    tri_prop = tri_prop.iloc[:, 1:]
    scaled_tri_prop = scale(tri_prop, axis=1) # Standardization
    tri_cols = tri_prop.columns.tolist()
    tri_prop = pd.DataFrame(scaled_tri_prop, columns=tri_cols)
    pp_tri = {}
    for i in range(64):
        key = tri_prop.columns[i]
        items = tri_prop.iloc[:, i].tolist()
        pp_tri[key] = items  
   
    seqLength = len(seq)
    sequence_vector = np.zeros([len(seq), 12])
    k = 3
    for i in range(seqLength - int(k) + 1):
        sequence_vector[i, 0:12] = pp_tri[seq[i:i+k]]
    return sequence_vector

def load_label_seq(seq_file):
    label_list = []
    fp = open(seq_file,'r',encoding='UTF-8')
    for line in fp:
        if line[0] == '>':
            name = line[1:-1]
            label = name.split(':')[-1]
            label_list.append(int(label))
    return to_categorical(np.array(label_list)) #返回二进制表示的标签

def kturple_frequency(seq_file):
    KNFP = []
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids() 
    
    fp = open(seq_file,'r',encoding='UTF-8')
    for line in fp:
        if '>' not in line:
            line = line.upper().strip()
            kmer1 = coden(line.strip(),1,tris1)
            kmer2 = coden(line.strip(),2,tris2)
            kmer3 = coden(line.strip(),3,tris3)
            Kmer = np.hstack((kmer1,kmer2,kmer3))
            KNFP.append(Kmer)
    return np.array(KNFP)

def Phy_property(seq_file):
    Phy_12 = []
    Phy_90 = []
    fp = open(seq_file,'r',encoding='UTF-8')
    for line in fp:
        if '>' not in line:
            line = line.upper().strip()
            probMatr_DPCP = noramlization(phy90(line))
            probMatr_DP12 = noramlization(phy12(line))
            Phy_12.append(probMatr_DP12)
            Phy_90.append(probMatr_DPCP)
    return np.array(Phy_12), np.array(Phy_90)

def load_Test_Data(seq_file_test,model):
    label = []
    fp = open(seq_file_test,'r',encoding='UTF-8')
    for line in fp:
        if line[0] != '>':
            line = line.upper().strip()
        elif line[0] == '>': 
            label.append(int(1))
    KNFP_Test = kturple_frequency(seq_file_test)
    Phy12,Phy19 = Phy_property(seq_file_test)
    Embedding_Test,  embedding_matrix = Generate_Embedding(seq_file_test, model)
    return to_categorical(np.array(label)), KNFP_Test, Phy12, Phy19, Embedding_Test, embedding_matrix

def to_matrix(seq_file):
    row_number = 81
    seq_data = []
    seq=[]
    #提取数据文件中的序列
    fp = open(seq_file,'r',encoding='UTF-8')
    for line in fp:
        if line[0]!='>':
            seq.append(line)
    for i in range(len(seq)):
        mat = np.array([0.] * 4 * row_number).reshape(row_number, 4)#(81*4) 零矩阵
        for j in range(len(seq[i])):

            if seq[i][j] == 'A':
                mat[j][0] = 1.0
            elif seq[i][j] == 'C':
                mat[j][1] = 1.0
            elif seq[i][j] == 'G':
                mat[j][2] = 1.0
            elif seq[i][j] == 'T':
                mat[j][3] = 1.0
        seq_data.append(mat)
    return np.array(seq_data)

def Trimer(seq_file):    
    lookup_table = []
    seq=[]
    #提取数据文件中的序列
    fp = open(seq_file,'r',encoding='UTF-8')
    for line in fp:
        if line[0]!='>':
            seq.append(line.strip())
    #for p in product('ATGC', repeat=3):
    for p in itertools.product('ATGC', repeat=3):
        w = ''.join(p)
        lookup_table.append(w)
    X2 = np.empty((len(seq), len(seq[0])-2))
    for i in range(len(seq)):
        for j in range(len(seq[0])-2):
            w = seq[i][j:j+3]
            X2[i,j] = lookup_table.index(w)      
    X2 = to_categorical(X2)
    X2=np.array(X2)#(20,79,64)
    return X2   

def load_Train_Data(seq_file, seq_model):
    Embedding,embedding_matrix = Generate_Embedding(seq_file, seq_model)
    label = load_label_seq(seq_file) #返回经过编码后的标签 
    KNFP = kturple_frequency(seq_file)
    Phy12,Phy19 = Phy_property(seq_file) 
    phy4 = to_matrix(seq_file)
    phy64 = Trimer(seq_file)
    phy64=np.pad(phy64,((0,0),(0,2),(0,0)),'constant',constant_values=0)
    return label, KNFP, Phy12, Phy19, Embedding, embedding_matrix,phy4,phy64
    
def seq2ngram(seqs, k, s, wv):
    list22 = []
    print('need to n-gram %d lines' % len(seqs))

    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line) 
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list22.append(convert_data_to_index(list2, wv))
    return list22
    
def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

def circRNA2Vec(k, s, vector_dim, seq_model, MAX_LEN, pos_sequences):
    from gensim.scripts.glove2word2vec import glove2word2vec
    
    model1 = KeyedVectors.load_word2vec_format(seq_model)    
    
    #model1 = gensim.models.Doc2Vec.load(seq_model)
    
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    seqs = pos_list 

    X = pad_sequences(seqs, maxlen=MAX_LEN,padding='post')

    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
            
    return X, embedding_matrix

def read_fasta_file(fasta_file):
    Lines = []
    fp = open(fasta_file,'r',encoding='UTF-8')
    for line in fp:
        if '>' not in line:
            line = line.upper().strip()
            Lines.append(line)
    return np.asarray(Lines)

def Generate_Embedding(ss, seq_model):
    seqpos = read_fasta_file(ss)
    X, embedding_matrix = circRNA2Vec(10, 1, 30, seq_model, 81, seqpos)
    return X, embedding_matrix

def defineExperimentPaths(basic_path, methodName, experimentID):
    experiment_name = methodName + '/' + experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    LOG_PATH = basic_path + experiment_name + '/logs/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    RESULT_PATH = basic_path + experiment_name + '/results/'
    mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    mk_dir(RESULT_PATH)
    mk_dir(LOG_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH]


def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)
    input_at = Activation('relu')(input_bn)
    input_dp = Dropout(0.3)(input_at)
    return input_dp

def ConvolutionBlock(input, f, k):
    A1 = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    A1 = bn_activation_dropout(A1)
    return A1 

def MultiScale(input):
    A = ConvolutionBlock(input,64,1)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 3)
    D = ConvolutionBlock(input, 64, 1)
    D = ConvolutionBlock(D, 64, 5)
    D = ConvolutionBlock(D, 64, 5) 
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('relu')(result)
    return result   

def createModel(embedding_matrix):
    filter_sizes = (2,3,4)
    num_filters = 32
    dropout_prob = (0.3, 0.8)
    conv_blocks = []

    phy4_input = Input(shape=(81, 4), name='phy4_input')
    phy4 = Convolution1D(filters=64,kernel_size=3,padding='same')(phy4_input)
    phy12_input = Input(shape=(81, 12), name='phy12_input')     
    phy12 = Convolution1D(filters=64,kernel_size=3,padding='same')(phy12_input)
    phy90_input = Input(shape=(81, 90), name='phy90_input')
    phy90 = Convolution1D(filters=64,kernel_size=3,padding='same')(phy90_input)
    phy64_input = Input(shape=(81,64),name='phy64_input')
    phy64 = Convolution1D(filters=64,kernel_size=3,padding='same')(phy64_input)
      
    ###################################################################################################################################
    embedding_input = Input(shape=(81, ), name='embedding_input')
    embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(embedding_input) 
    embeddings = Dropout(0.3)(embeddings)
    embedding=MultiHeadAttention(head_num=6,name='Multi-Head')(embeddings)   
    #embedding = Convolution1D(filters=64,kernel_size=3,padding='same')(embedding)
    #embedding = bn_activation_dropout(embedding)
    #embedding = Convolution1D(filters=32,kernel_size=2,padding='same')(embedding)
    #embedding = bn_activation_dropout(embedding)     
    #
    #embedding=Add()([merge,embedding])
    #残差块
    #
    #phy2 = Convolution1D(filters=81,kernel_size=7,padding='valid')(embedding)
    #phy2 = bn_activation_dropout(phy2)
    #phy2 = AveragePooling1D(pool_size=5)(phy2)
    #phy2 = bn_activation_dropout(phy2)
    #phy2 = Bidirectional(LSTM(81,return_sequences=True))(phy2)#120
    #phy2 = Flatten( )(phy2)       
  
        
    merge0 = Concatenate( )([phy4, phy12, phy90,phy64])
 #   phy1 = MultiScale(merge0)
    
    phy1 = Convolution1D(filters=256,kernel_size=5,padding='same')(merge0)
    phy1 = bn_activation_dropout(phy1)
    phy1 = Convolution1D(filters=128,kernel_size=4,padding='same')(phy1)
    phy1 = bn_activation_dropout(phy1)
    phy1 = Convolution1D(filters=64,kernel_size=3,padding='same')(phy1)
    phy1 = bn_activation_dropout(phy1)
    phy1 = Convolution1D(filters=32,kernel_size=2,padding='same')(phy1)
    phy1 = bn_activation_dropout(phy1)   
    
    mergeInput2 = Concatenate()([embedding,phy1])
    #多尺度卷积
  #  Merge = MultiScale(mergeInput2)
  #  phy2 = Convolution1D(filters=256,kernel_size=5,padding='same')(mergeInput2)
    #phy2 = bn_activation_dropout(phy2)
    #phy2 = AveragePooling1D()(phy2)
    #phy2 = Convolution1D(filters=128,kernel_size=4,padding='same')(phy2)
    #phy2 = bn_activation_dropout(phy2)
    #phy2 = Convolution1D(filters=64,kernel_size=3,padding='same')(phy2)
    #phy2 = bn_activation_dropout(phy2)  
    
    mergeInput3 = Flatten( )(mergeInput2)
    overallResult = Dense(512,activation='relu')(mergeInput3)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = Dense(256, activation='relu')(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = Dense(128, activation='relu')(overallResult)
    overallResult = Dropout(0.3)(overallResult)    
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)
    return Model(inputs=[phy4_input, phy12_input, phy90_input, embedding_input,phy64_input], outputs=[ss_output])    


def transfer_label_from_prob(proba):
    label = [0 if val <= 0.5 else 1 for val in proba]
    return label

def calculate_performance(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
#    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc,sensitivity, specificity, MCC

if __name__ == "__main__":
    
    #############sigma70
    gpu_id='0'
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')
    tf_config=tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    tf.compat.v1.Session(config=tf_config)    
    basic_path = './RESULT1_best/'
    methodName = 'Caps'

    batchSize = 60
    maxEpochs = 200    
    #pos = [0,1] neg = [1,0]

    logging.basicConfig(level=logging.DEBUG)
    sys.stdout = sys.stderr
    logging.debug("Loading data...")
                  
 
    seq_file = './data0.fasta'
   # seq_file_test = './independent.txt'
    seq_model = './DNAVectors.txt'
    #model = '/home/wangyansong/Result/OnlyStaticNLP/Doc2Vec/circRNA2Vec_model'
    
    Label_Train, KNFP, Phy12, Phy90, Embedding_Train, embedding_matrix ,phy4,phy64 = load_Train_Data(seq_file, seq_model)
    
    #Label_Test, KNFP_Test, Phy12_Test, Phy90_Test, Embedding_Test, embedding_matrix = load_Test_Data(seq_file_test, model)
    
    #test_y = Label_Test[:, 1]
    
    i = 0
    #aucs = []
    Acc = []
    Sensitivity = []
    Specificity = []
    MCC = []    
    #kf = KFold(5, True)
    #print(KNFP.shape[0])
    indices = np.random.permutation(phy4.shape[0])
    training_idx, test_idx = indices[:4576], indices[4576:] #划分训练测试集数据以及标签
    train_X1, eval_X1 = phy4[training_idx, :, :], phy4[test_idx, :, :]
    train_X2, eval_X2 = Phy12[training_idx, :, :], Phy12[test_idx, :, :]
    train_X3, eval_X3 = Phy90[training_idx, :, :], Phy90[test_idx, :, :]
    train_X4, eval_X4 = Embedding_Train[training_idx, :], Embedding_Train[test_idx, :]
    train_X5, eval_X5 = phy64[training_idx, :, :], phy64[test_idx, :, :]
    
    #train_X6, eval_X6 = phy64[training_idx, :, :], phy64[test_idx, :, :]
    
    train_y, eval_y = Label_Train[training_idx, :], Label_Train[test_idx, :] 
    for k in range(1,6):
        
        [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH] = defineExperimentPaths(basic_path, methodName,
                                                                                         str(i))
        logging.debug("Loading network/training configuration...")
        model = createModel(embedding_matrix)
        #model.load_weights('./YYYYY/model0_result/Caps/0/checkpoints/weights.best.hdf5')
        logging.debug("Model summary ... ")
        model.count_params()
        model.summary()
        checkpoint_weight = CHECKPOINT_PATH + "weights.best.hdf5"
        if (os.path.exists(checkpoint_weight)):
            print ("load previous best weights")
            model.load_weights(checkpoint_weight)
        #adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer='Adam',############'rmsprop'Adam RMSprop
                    #loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,metrics = ['accuracy'])##'ss_output': 'categorical_crossentropy'
                        loss={'ss_output': 'binary_crossentropy'},metrics = ['accuracy'])
        logging.debug("Running training...")

        def step_decay(epoch):
            initial_lrate = 0.001
            drop = 0.8
            epochs_drop = 5.0            
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            print (lrate)
            return lrate
         
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, verbose=2, mode='auto'),
            ModelCheckpoint(checkpoint_weight,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='auto',
                            period=1),
            #LearningRateScheduler(step_decay),
            ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)]        
            #TensorBoard(log_dir=LOG_PATH, histogram_freq=1, write_graph=True, write_images=True,
                                                            #embeddings_freq = 1,
                                                            #embeddings_layer_names = None,
                                                            #embeddings_metadata = None)]
        startTime = time.time()
        history = model.fit(
            {'phy4_input': train_X1, 'phy12_input': train_X2, 'phy90_input': train_X3, 'embedding_input': train_X4,'phy64_input':train_X5},
            {'ss_output': train_y},
            epochs=maxEpochs,
            batch_size=batchSize,
            callbacks=callbacks,
            verbose = 1,
            validation_data=(
                {'phy4_input': eval_X1, 'phy12_input': eval_X2, 'phy90_input': eval_X3, 'embedding_input': eval_X4,'phy64_input':eval_X5},
                {'ss_output':eval_y}),
            shuffle=True)
        endTime = time.time()
        logging.debug("Saving final model...")
        model.save(os.path.join(MODEL_PATH, 'model.h5'), overwrite=True)
        json_string = model.to_json()
        with open(os.path.join(MODEL_PATH, 'model.json'), 'w') as f:
            f.write(json_string)
        logging.debug("make prediction")       
        #ss_y_hat_test = model.predict(
               #{'sequence_input': KNFP_Test, 'phy12_input': Phy12_Test, 'phy90_input': Phy90_Test, 'embedding_input': Embedding_Test}) 
        ss_y_hat_test = model.predict({'phy4_input': eval_X1, 'phy12_input': eval_X2, 'phy90_input': eval_X3, 'embedding_input': eval_X4,'phy64_input':eval_X5}) 
        
        ypred = transfer_label_from_prob(ss_y_hat_test[:, 1])
 
        
        test_y = eval_y[:, 1]
        ytrue = test_y
        #ypred = ss_y_hat_test[:, 1]
    
        y_pred = np.argmax(ss_y_hat_test, axis=-1)
        
        acc, sensitivity, specificity, mcc = calculate_performance(len(ytrue), ypred, ytrue)
    
        #auc = roc_auc_score(ytrue, ypred)
        #print(auc)
        #fpr,tpr,thresholds=roc_curve(ytrue,ypred)
        #tprs.append(interp(mean_fpr,fpr,tpr))
        #aucs.append(auc)        
        #acc = accuracy_score(ytrue, y_pred)
        np.save(RESULT_PATH + 'acc.npy',acc)
        print(acc)
        Acc.append(acc)
        
        #precision = precision_score(ytrue, y_pred)  
        #recall = recall_score(ytrue, y_pred)
        #fscore = f1_score(ytrue, y_pred)         
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)
        MCC.append(mcc)     
        i = i + 1
        k = k + 1
    

    #print("acid AUC: %.4f " % np.mean(aucs))
    print(' ACC values of five fold')
    print(Acc)    
    print("acid ACC: %.4f " % np.mean(Acc))
    
    print('Sensitivity values of five fold')
    print(Sensitivity)    
    print("acid sensitivity: %.4f " % np.mean(Sensitivity))
     
    print('specificity values of five fold')
    print(specificity)      
    print("acid specificity: %.4f " % np.mean(Specificity))
    
    print('MCC values of five fold')
    print(MCC)    
    print("acid MCC: %.4f " % np.mean(MCC))
    
    mean_acc = np.mean(Acc)

    np.save(basic_path + methodName + '/' + 'mean_acc.npy',mean_acc)
    
    
  
