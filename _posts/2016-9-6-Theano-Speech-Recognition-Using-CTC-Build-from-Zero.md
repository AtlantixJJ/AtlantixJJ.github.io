---
layout: post
title:  "Build a CTC Network on Theano for Speech Recognition from Scratch (Unfinished) "
date:   2016-09-07 17:03:23 +0800
categories: jekyll update
---

## Build a CNN for Speech Recognition
To build a CNN for speech recognition have two difficulties:
+ How to adapt to different input length.
+ How to transform network output to framewise prediction.
I don't know if I should or should not assume a fixed-length input. Most RNN-based sequence processor use input of the same length, otherwise batch training is impossible . They make it possible by padding input sequence to the same length while using `mask` to block padded information. Well, I know I am going to put recurrent layers on the top of CNN, so I don't know if enabling non-fixed input length is useful in training. But in practical application, I think I should provide an elegant solution for varying input by enabling network with arbitrary input length.
So the first problem I want to solve is to build a CNN capable of arbitrary input length.

### A forward-only CNN
{%highlight python%}
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

This code passed tests. If I build the network output as function and pass it some values, it will get a right-shaped output and some random number. Seems to be good : ).
The next step is to enable CTC loss function. Well we can start framewise network building right now, but don't forget out target is to enable CTC training.

Here I just use keras CTC loss function. And the added lines is shown below:

{%highlight python%}

self.raw_out = self.conv_layers[-1].out[:,:,:,0]
# we start from here
self.softmax_layer = MSoftmax(self.raw_out,ifTimeDist=True)
self.pred = self.softmax_layer.softmax_output

# CTC batch Loss layer construction
self.ctc_input_len = input_length
self.ctc_label_len = label_length
self.ctc_label = ctc_labels
self.loss = T.mean( K.ctc_batch_cost(self.ctc_label,
  self.pred,self.ctc_input_len,self.ctc_label_len) )

# Prepare For Differentiation
self.input_vars = [ self.input,self.ctc_label,
  self.ctc_input_len,self.ctc_label_len ]

self.params = [layer.W for layer in self.conv_layers]
self.params += [layer.b for layer in self.conv_layers]
{% endhighlight %}
 There is a small tactic: use `T.mean`. Other wise you will get a loss for each examples in a batch, while cost must be a scalar. If you test the `loss` as network function's output, you will get a number around 1400, which is also the loss if you train a CTC network using keras approximately. By the way, you should set the variables like this:

{%highlight python%}
 train_x = T.ftensor4("train_x")
input_length = T.matrix("CTC_Input_Length",dtype="int32") # notice: matrix rather than vector
label_length = T.matrix("Label_Length",dtype="int32") # notice: matrix rather than vector
ctc_labels = T.matrix("CTC_Labels",dtype="int32")
Net = Base.MConvNet(train_x,ctc_labels,input_length,label_length,input_shape=None)

# for testing loss functions
i = 0
bi = i * batch_size
ei = bi + batch_size
train_data = feature[bi:ei,:,:,:]
train_y = np.zeros((batch_size,maxil),dtype="int32")
len_y = np.array([len(sent[i+bi]) for i in range(ei-bi)],dtype="int32")
len_pred = np.array([399 for i in range(batch_size)],dtype="int32")
for j in range(batch_size):
    train_y[j,:len_y[j] ] = sent[bi+j][:].astype("int32")

res = Net.loss.eval( {train_x:train_data,
    ctc_labels:train_y,
    label_length:np.expand_dims(len_y,1),
    input_length:np.expand_dims(len_pred,1)})

{% endhighlight %}

 Well you may notice in the loss function testing part, you cannot do as I propose. Because I am experimenting on TIMIT dataset and transformed the dataset. So it is of no use even if I post the entire code : ).
 Now we are going to differentiate the network. In this step, I apply `Adadelta` and reuse `Deep Learning Tutorial's lstm.py`. I wrap the process of compilation into a class, to which I may add more optimizers. ( Am I going to write a ,eh.., something like keras in the future? )

{%highlight python%}
 class NetworkCompiler(object):
    """
    Compile the network given optimizer and theano model
    """
    def __init__(self,model,optimizer='Adadelta'):
        """
        @param model Theano model
        @param optimizer 'Adam' 'Adadelta' 'SGD' ...
        """
        self.model = model
        if optimizer == 'Adadelta':
            self.grad_func,self.update_func = adadelta(1,
                self.model.params,self.model.input_vars,
                self.model.loss)
{% endhighlight %}

Personally I am really confused with the original `Adadelta`code. Schematically, this function produces two functions:
+ f_grad_shared: Takes in all inputs, produce an output, and prepare the update values.
+ f_update: do updates (apply update values to parameters)

But I am rather confused in two ways:
+ tparams is not theano variable but dictionary instead
+ does lr has any effect? What's more, I think `Adadelta` prefers lr to be set to 1, having nothing to do with learning rate scheduling.

{%highlight python%}
def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameters
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update
{% endhighlight %}

Here is my modified version. My modification includes:
+ Switch dictionary parameter to ordinary theano shared variable parameter.
+ Enable multiple inputs by using a list of shared varibles.
+ Removed `lr`.

{%highlight python%}
def adadelta(tparams,inputs, cost):
    """
    tpramas: Theano SharedVariable
        Model parameters
    inputs: list of inputs (including x & y)
    cost: Theano variable
        Objective fucntion to minimize
    """
    grads = [T.grad(cost,param) for param in tparams]

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.))
                    for p in tparams]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.) )
                   for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.))
                for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up)

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams, updir)]

    f_update = theano.function([], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update
{% endhighlight %}

Anyway, this compiles. Now it comes to train the network. To obtain a more tractable training procedure, I wrote a class `BatchTrainingScheduler`, responsible for preparing batch data, training and saving weights.
