---
layout: post
title:  "Keras Learning Note: Framewise RNN Setup and Convolution integration"
date:   2016-08-18 16:03:23 +0800
categories: jekyll update
---

## Step-by-step introduction of Framewise RNN in Keras

Take Speech Recognition as example, RNN takes in frame input and produce framewise output. Using keras, this can be achieved with minimal code. But there is a shortcoming of using keras, that all input frame need to be of the same length to enable batch training. You will have to use `keras.preprocessing.sequence.pad_sentence` method to get your data ready for batch training. If you are disgusted by padding, you will get to know how to train with `batchsize=1` without padding at the end of this blog.

### Data Preparation

First of all, let's get onto data preparation. Well, I will take Speech Recognition as example again. Now you have a `.wav` audio file which you have loaded into memory and converted into frequency spectrum. ( You can refer to [python speech feature](https://github.com/jameslyons/python_speech_features) if you don't know how to complete the step above.) The spectrum is a 2-dim array which is expected to have the shape `(FrameLength,FeatureSize)`. Using keras' interface make it easy to adjust the length of training sequences.

{% highlight python %}
# Train_data is a list of spectrums
from keras.preprocessing import sequence
Train_data = sequence.pad_sentence(Train_data)
# Now Train_data is 3D numpy ndarray type
{% endhighlight %}

 Keras' `models.fit` method expect training data with format `numpy.ndarray` and shape `(sample,length,feature)`. As for label, in case you want to use `cross_entropy` loss, you will need to transform the label into  probability-distribution-like target like the figure below.
![Segmentation]({{ site.baseurl }}/assets/seg.png)

Now the data preparation is done! If you have your model properly built, you can then call fit method to train your net.
So let's get onto the network building. Give a glance at keras' LSTM examples and you will find that similar code won't work for framewise predictions.
For example, `imdb_lstm.py`, the simplest one:
{% highlight python %}
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))
{% endhighlight %}
IMDB is a widely used dataset for various purpose. Here it is used to classify emotion. So the example takes in a sequence and produce a single scalar as output. As for `Embedding`, it is a layer for transforming a scalar presentation (e.g. no. of vocabulary) into another vector representation with different dimension. This example looks quite simple but it is far from what we need to handle.
Just think about it, how can a normally placed `Dense` layers be aware of the time axis?
So the trick here is to use `layers.TimeDistributed` on Dense layers. Like this:
{% highlight python %}
model.add(TimeDistributed(Dense(n_class)))
{% endhighlight %}
As expected, our simple net compiles.

### Make your model deep and bi-directional
You may consider stacking RNN or LSTM. And make bi-directional RNNs, which is quite popular in recent literatures.
To enable bi-directional RNN is quite simple. Just initialize with `go_backwards=True` and stack the layers together. Keras' example code gives quite a clear example:

{% highlight python %}
forwards = LSTM(64)(embedded)
# apply backwards LSTM
backwards = LSTM(64, go_backwards=True)(embedded)
# concatenate the outputs of the 2 LSTMs
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
# or mode='sum'
after_dp = Dropout(0.5)(merged)
output = Dense(1, activation='sigmoid')(after_dp)
{% endhighlight %}

But there is a pitfall with deep bi-directional networks. You have to enable `return_sequences` or Recurrent layer will simply return single result rather than framewise output. In addition, one problem confused me for a long time. Whether shall we enable `go_backwards` is deep backward RNN?

{% highlight python %}
forward1 = layers.LSTM(1024,return_sequences=True)(Input)
forward2 = layers.LSTM(1024,return_sequences=True)(forward1)
backward1 = layers.LSTM(1024,go_backwards=True,return_sequences=True)(Input)
# Aha, which one should you choose?
if bw == True:
    backward2 = layers.LSTM(1024,go_backwards=True,return_sequences=True)(backward1)
else:
    backward2 = layers.LSTM(1024,return_sequences=True)(backward1)
{% endhighlight %}

If you dive into keras' code, you will find the related core source code is in `theano_backend.py`. Let's take a close look at the the backwards related part. Notice that this part contains some duplicated code for different situations like mask or unroll, which is surely identical in terms of backwards computation.

{% highlight python %}
def _step(input, *states):
    output, new_states = step_function(input, states)
    return [output] + new_states

results, _ = theano.scan(
    _step,
    sequences=inputs,
    outputs_info=[None] + initial_states,
    non_sequences=constants,
    go_backwards=go_backwards)
    # giving backwards to theano.scan make the input sequence just the reverse.

# deal with Theano API inconsistency
if type(results) is list:
    outputs = results[0]
    states = results[1:]
else:
    outputs = results
    states = []

outputs = T.squeeze(outputs)
last_output = outputs[-1]

axes = [1, 0] + list(range(2, outputs.ndim))
outputs = outputs.dimshuffle(axes)
states = [T.squeeze(state[-1]) for state in states]
return last_output, outputs, states
{% endhighlight %}

So now we can see that keras did not reverse the output sequence. As suggest by https://github.com/fchollet/keras/issues/3448, we need to add reverse layer manually.
Alternatively, we can stack two `go_backwards` together...
Adding `go_backwards` or not does not affect the computational efficiency, since this operation is simply a twist of input! For deep RNNs, they always complete a whole layer's computation before they progress into deeper ones. Gradient computation is similar to this. As a result, the training speed of both versions is almost the same, which I have confirmed on TIMIT dataset.  

In addition, I have to say that this version of deep bi-directional RNN is not a popuplar one. Rather I would like to call it bi-directional deep RNN. Most DBRNN combine their output after one layer, which mixes forward information and backward information quickly. While I propose that stacking deeper before forward and backward information mixes is prone to improve representational power of uniform direction. And we just need to combine forward and backward feature with a shallow architecture. In this way, the network has a clearer structure.



## Incorporate Convolution with LSTM
Using RNN only is rather simple as it is like utilizing a standard model. But adding a convolution layer requires more techniques and tries.
To achieve this, you have a bunch of tips to know in priority (which takes tens of tries for me to realize them):
+ Convolution assumes a 3D input shape setting. ( Maybe 4D actually). So input shape is fixed in any way.
+ Though input shape is fixed, parameters can be loaded (finetuned) for different input shape. This make it possible for changing length.
+ Between convolution output and LSTM input, you need to `reshape` and `permute`.
+ Be really careful about output shapes. You have to compute reshape's shape, convolution's input and output shape manually.

{% highlight python %}
"""
@param maxif the max length of input frame
@param convf_num Convolution filter number of the first layer. But I set each deeper layer is double the size of the previous one
@param conv_t time axis' expand of convolution
@param conv_f frequency axis' expand of convolution
"""
FrameInput = layers.Input(shape=(3,maxif,featuresize),dtype='float32',name="Framewise_Input")

Conv1 = layers.Convolution2D(conv_fnum,conv_t,conv_f, border_mode='valid',
activation="tanh", input_shape=(3,maxif,featuresize), name='conv1')(FrameInput)
len2 = maxif - conv_t + 1
h2 = featuresize - conv_f +1
Conv2 = layers.Convolution2D(conv_fnum * 2,conv_t,h2,border_mode='valid',
activation='tanh',input_shape=(conv_fnum,len2,h2),name='conv2')(Conv1)
# output shape: (samples,conv_fnum * 2, len2 -conv_t+1 ,1)

rnn_shape = ( conv_fnum * 2 , len2 - conv_t +1)
# input_len is the input length of CTC, i.e.
self.input_len = rnn_shape[1]
# The main role of reshape is to change 3D output into 2D input
Reshape_1 = layers.Reshape(target_shape=rnn_shape,name="Reshape_1")(Conv2)
# Permute is used to switch axis.
Permute_1 = layers.Permute(dims=(2,1),name="Permute_1")(Reshape_1)
DenseToRNN = layers.TimeDistributed(layers.Dense(conv_fnum*4))(Permute_1)

FLSTM_1 = layers.LSTM(conv_fnum*4,return_sequences=True,name='Forward_LSTM_1')(DenseToRNN)
FLSTM_2 = layers.LSTM(conv_fnum*4,return_sequences=True,name='Forward_LSTM_2')(FLSTM_1)
BLSTM_1 = layers.LSTM(conv_fnum*4,go_backwards=True,return_sequences=True,name="Backward_LSTM_1")(DenseToRNN)
BLSTM_2 = layers.LSTM(conv_fnum*4,go_backwards=True,return_sequences=True,name="Backward_LSTM_2")(BLSTM_1)
merge1 = layers.merge([FLSTM_2,BLSTM_2],mode='concat')
dense = layers.TimeDistributed( layers.Dense(n_phonemes+1) )(merge1)

FramePred = layers.Activation('softmax',name='Softmax')(dense)
{% endhighlight %}

Well, I think I make it quite clear in the comments. In summary, the main difficulties for new beginners is just the four points mentioned above.
I have trained my neural network on TIMIT, but it seemed quite strange that it overfits heavily. On training set, I achieved frame accuracy of 95% but on core test set it was only 45%. Even if I cut down the network to 32 convolution filters ( 128 for LSTMs), the result was almost the same. Well, for overfitting problems, one is expected to add weight regularization terms and enlarge training set. So I applied the following enhancement:
![Concatenated Input data]({{site.baseurl}}/assets/TIMIT_concate.png)
+ Enlarged the dataset for almost one time larger by downsampling the wav files (from 16000K to 8000K). For human ears, the downsampled voice become a little bit vague but it is recognizable.
+ Stochastically concatenated utterances.
The code is running on GPUs right now. And a few days later you can see the report.

To sum up, this is how I setup a framewise RNN integrated with CNN. In those days, I really had a hard time figuring out problems which seems nothing to me right now. If I have any mistake, please point it out for me, as I am no more than a new learner.  Hope this blog can help you save the time that I once wasted.

Please give the credit to the original author if you use it elsewhere.

Jianjin, Xu.
Department of Computer Science, Tsinghua University.
