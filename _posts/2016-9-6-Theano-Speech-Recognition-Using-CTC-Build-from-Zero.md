

In this blog, I will tell you the process of my setup of a CTC enabled CNN-LSTM hybrid neural network on the top of Theano. I will show you the essential part of implementation, skipping some data-relevant code. Full code may be available on my repository later.

## Build a CNN for Speech Recognition
To build a CNN for speech recognition have two difficulties:

+ How to adapt to different input length.
+ How to transform network output to framewise prediction.

I don't know if I should or should not assume a fixed-length input. Most RNN-based sequence processor use input of the same length, otherwise batch training is impossible . They make it possible by padding input sequence to the same length while using `mask` to block padded information. Well, I know I am going to put recurrent layers on the top of CNN, so I don't know if enabling non-fixed input length is useful in training. But in practical application, I think I should provide an elegant solution for varying input by enabling network with arbitrary input length.

So the first problem I want to solve is to build a CNN capable of arbitrary input length.

### A forward-only CNN

{%highlight python%}

class MSoftmax(object):
    def __init__(self,input,ifTimeDist=False):
        if ifTimeDist:
            # scan is for time distributed computation.
            # softmax and scan seems to ignore batch axis (0).
            # input shape: (batch_size,128,401)
            self.softmax_output,self.update = theano.scan(T.nnet.softmax,sequences=[input.dimshuffle(0,2,1)])
            # self.pred works. result (batch_size,401,128) [i,j,:] sums to 1
        else:
            self.softmax_output = T.nnet.softmax(input)

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

Personally, I don't know how to produce a framewise probability distribution elegantly. Now the only way that I came up with is to scan through all time steps... Maybe doing softmax along time axis shouldn't bother to use scan ? I would be appreciative if you can tell me how to do.

Anyway, this code passed tests. If I build the network output as function and pass it some values, it will get a right-shaped output and some random number. Prediction layer will get each time step's prediction for each example sums up to 1. Seems to be good : ).

### Enable Training

The next step is to enable CTC loss function. Here I just use keras CTC loss function. And the added lines is shown below:

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

 Well you may notice in the loss function testing part, you cannot do as I suggest. Because I am experimenting on TIMIT dataset and I have transformed the dataset. So it is of no use even if I post the entire code : ). Here I am just showing you the core implementation of a CTC network.

 Now we are going to differentiate the network. In this step, I apply `Adadelta` and reuse `Deep Learning Tutorial's lstm.py`. I wrap the process of compilation into a class, to which I may add more optimizers. ( Am I going to write a ,eh.., another keras in the future? )

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

But I am rather confused in two ways:

+ tparams is not theano variable but dictionary instead
+ does lr has any effect? What's more, I think `Adadelta` prefers lr to be set to 1, having nothing to do with learning rate scheduling.

Again, please tell me if you know why : ) .

Here is my modified version. Modification includes:

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

Anyway, this compiles. Now it comes to train the network. To obtain a more tractable training procedure, I wrote a class `BatchTrainingScheduler`, responsible for preparing batch data, training and saving weights. Again, this evolves much unnecessary details, so I skip the implementation here.

In addition, I added framewise loss for comparation. This requires a new layer implementation:

{%highlight python%}
class MCrossEntropyFromSoftmax(object):
    def __init__(self,softmax_output,label_prob,ifTimeDist=False):
        if ifTimeDist:
            # softmax_output is expect to be (batch,401,128)
            self.frame_log,self.update = theano.scan(T.log,sequences=[softmax_output])
            self.frame_loss = - T.mean( T.mean( T.sum( label_prob * self.frame_log,axis=2) ,axis=0) ,axis=0)
            # loss shape : (401)
        else:
            self.frame_loss = - T.mean(label_prob * T.log(softmax_output))
{% endhighlight %}

Now the CTC-CNN runs.

{%highlight python%}
Using gpu device 2: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Using Theano backend.
Loading Training Data:
('./data/CoreTest_Audio_3.npy', './data/CoreTest_Label_3.npy', './data/CoreTest_Sentence_3.npy')
Dataset loading compelte.
Max Frame: 405 	 Max Sequence Length: 67
Label length: 399
Building Model
Using base convolution filter: None
Model name: THEANO
Building model...
Compiling for CTC...
[audio_input, CTC_Labels, CTC_Input_Length, Label_Length]
Compiling for Framewise Prediction...
[audio_input, audio_label]
Checking Network Output...
(399, 40)
Checking Frame Loss...
3.68886995316
Checking CTC Loss...
Train using CTC loss ...
1266.20937281
1266.20937281
1280.3120997
1084.664638
786.608295334
712.46418894
708.009468547
709.193808295
699.881733037
683.349069459
678.93139635
679.363749416
679.713211679
Train using Framewise Loss...
3.82449817657
3.77304577827
3.27461051941
2.84103941917
2.87398314476
2.90601921082
2.94813394547
2.85982584953
2.88489484787
2.98423242569
2.90129828453
2.98246645927
Train using CTC Loss...
696.61584301
695.274994058
695.945729375
701.028371696
698.802302323
...
{%endhighlight%}

Aha, things goes fine. Em.. Don't you think training with two loss function is great?


## Adding LSTM !


---
layout: post
title:  "Build a CTC Network on Theano for Speech Recognition from Scratch (Unfinished) "
date:   2016-09-07 17:03:23 +0800
categories: jekyll update
---
