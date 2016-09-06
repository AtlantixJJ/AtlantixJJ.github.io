---
layout: post
title:  "Build a CTC Network on Theano for Speech Recognition from Scratch (Unfinished) "
date:   2016-09-7 17:03:23 +0800
categories: jekyll update
---

## Build a CNN for Speech Recognition
To build a CNN for speech recognition have two difficulties:
+ How to adapt to different input length.
+ How to transform network output to framewise prediction.
I don't know if I should or should not assume a fixed-length input. Most RNN-based sequence processor use input of the same length, otherwise batch training is impossible. They make it possible by padding input sequence to the same length while using `mask` to block padded information. Well, I know I am going to put recurrent layers on the top of CNN, so I don't know if enabling non-fixed input length is useful in training. But in practical application, I think I should provide an elegant solution for varying input by enabling network with arbitrary input length.
So the first problem I want to solve is to build a CNN capable of arbitrary input length.

### A forward-only CNN
{ % highlight python %}
class MConv2D(object):
    """
    My 2D convolution layer. Responsible for forward building.
    """
    def __init__(self,input,filter_shape):
        """
        @param input tensor of input
        @param filter_shape 4-dim tuple (n_filters,channel,height,width)
        """
        self.input = input
        # Initialize weight
        self.filter_shape = filter_shape
        self.W = theano.shared(init_conv_weight(self.filter_shape,(1,1)))
        self.b = theano.shared(np.zeros(
            shape=(self.filter_shape[:1]),dtype=theano.config.floatX))

        self.z = self.b.dimshuffle(0,'x','x') + conv2d(
            input=input,filters=self.W,filter_shape=self.filter_shape,
            border_mode='valid')

        self.out = T.tanh(self.z)


class MConvNet(object):
    """
    Give a forward expression of CNN.
    """
    def __init__(self,input,base_num=64,stride=3,n_class=39):
        self.input = input
        self.n_class = n_class
        self.conv_shape = [(base_num,3,stride,30),
        (base_num*2,base_num,stride,11),
        (n_class,base_num*2,stride,1)]

        self.conv_layers = [MConv2D(input,self.conv_shape[0])]
        for i in range(len(self.conv_shape)-1):
            self.conv_layers.append(
                MConv2D(self.conv_layers[i].out,self.conv_shape[i+1]) )

        # final layer's shape:(batch_size,diminished_length,n_class,1)
        self.raw_out = self.conv_layers[-1].out[:,:,:,0]
{% endhighlight %}
