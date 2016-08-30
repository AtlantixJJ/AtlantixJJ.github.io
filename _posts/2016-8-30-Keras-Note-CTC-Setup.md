---
layout: post
title:  "Keras Learning Note: Speech Recognition CTC Setup"
date:   2016-08-30 16:03:23 +0800
categories: jekyll update
---

## Running CTC Network using keras
Keras has just merged a request containing batch CTC loss function. Although it contains an example code, but it is an image OCR program, and it is quite hard to read... This new coming functionality does not have a good documentation and code example, having costed me nearly two week before get my CTC network to working. Well I will first show how I achieve this.
Network definition: ( You can read the code beginning from `FrameInput`)

{% highlight python %}
class ConvCTCDiLSTM(object):
    def __init__(self,name="ConvCTCLSTM",maxif=405,maxil=74,conv_fnum=128,conv_t=3,conv_f=30,pool_t=2,pool_f=1,n_phonemes=39,featuresize=40,weight_path=None,ifFT=False,
    optimi='Adam',loss='categorical_crossentropy',metric=['accuracy']):
        self.optim = optimi
        self.name = name
        self.maxif = maxif
        self.maxil = maxil

        labels = layers.Input(shape=[maxil],dtype='float32',name="CTC_Label")
        # Input to CTC: the network's prediction length
        input_length = layers.Input(shape=[1],dtype='int32',name="input_length")
        # Input to CTC: length of sequence.
        label_length = layers.Input(shape=[1],dtype='int32',name="label_length")
        # Original structure
        FrameInput = layers.Input(shape=(3,maxif,featuresize),dtype='float32',name="Framewise_Input")

        Conv1 = layers.Convolution2D(conv_fnum,conv_t,conv_f, border_mode='valid',
        activation="tanh", input_shape=(3,maxif,featuresize), name='conv1')(FrameInput)
        len2 = maxif - conv_t + 1
        h2 = featuresize - conv_f +1
        Conv2 = layers.Convolution2D(conv_fnum * 2,conv_t,h2,border_mode='valid',
        activation='tanh',input_shape=(conv_fnum,len2,h2),name='conv2')(Conv1)
        # output shape: (samples,conv_fnum * 2, len2 -conv_t+1 ,1)
        rnn_shape = ( conv_fnum * 2 , len2 - conv_t +1)
        self.input_len = rnn_shape[1]
        Reshape_1 = layers.Reshape(target_shape=rnn_shape,name="Reshape_1")(Conv2)
        Permute_1 = layers.Permute(dims=(2,1),name="Permute_1")(Reshape_1)
        DenseToRNN = layers.TimeDistributed(layers.Dense(conv_fnum*4))(Permute_1)

        FLSTM_1 = layers.LSTM(conv_fnum*4,return_sequences=True,name='Forward_LSTM_1')(DenseToRNN)
        #FLSTM_2 = layers.LSTM(conv_fnum*4,return_sequences=True,name='Forward_LSTM_2')(FLSTM_1)
        BLSTM_1 = layers.LSTM(conv_fnum*4,go_backwards=True,return_sequences=True,name="Backward_LSTM_1")(DenseToRNN)
        #BLSTM_2 = layers.LSTM(conv_fnum*4,go_backwards=True,return_sequences=True,name="Backward_LSTM_2")(BLSTM_1)
        merge1 = layers.merge([FLSTM_1,BLSTM_1],mode='concat')
        dense = layers.TimeDistributed( layers.Dense(n_phonemes+1) )(merge1)

        FramePred = layers.Activation('softmax',name='Softmax')(dense)
{% endhighlight %}
