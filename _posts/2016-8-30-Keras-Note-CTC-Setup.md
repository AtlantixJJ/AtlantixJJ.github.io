---
layout: post
title:  "Keras Learning Note: Speech Recognition CTC Setup"
date:   2016-08-30 16:03:23 +0800
categories: jekyll update
---

## Running CTC Network using keras
Keras has just merged a request containing batch CTC loss function. Although it contains an example code, but it is an image OCR program, and it is quite hard to read... This new coming functionality does not have a good documentation and code example, having costed me nearly two week before get my CTC network to working. Well now I will not bother to mention those unhappy days.

Network definition: ( You can read the code beginning from `FrameInput`, and you most essential part starts from `ctc_loss_out`)

{% highlight python %}
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class ConvCTCDiLSTM(object):
    """
    Enabling CTC training on convolution and Deep bidirectional LSTM network
    """
    def __init__(self,name="ConvCTCLSTM",maxif=405,maxil=74,conv_fnum=128,conv_t=3,conv_f=30,pool_t=2,pool_f=1,n_phonemes=39,featuresize=40,weight_path=None,ifFT=False,
    optimi='Adam',loss='categorical_crossentropy',metric=['accuracy']):
        """
        @param maxif the max length of input frame
        @param maxil the max length of phoneme sequence
        @param convf_num Convolution filter number of the first layer. But I set each deeper layer is double the size of the previous one
        @param conv_t time axis' expand of convolution
        @param conv_f frequency axis' expand of convolution
        """
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

        ctc_loss_out = layers.Lambda(ctc_lambda_func,output_shape=(1,),name="CTC_Loss")([FramePred,labels,input_length,label_length])
        self.keras_model = models.Model(input=[FrameInput,labels,input_length,label_length],output=[ctc_loss_out])

        model_def = self.keras_model.to_json()
        file = open(self.name+"\_model_def.json","w")
        file.write(model_def)
        file.close()

        if weight_path is not None:
            print("Loading model from file.")
            self.keras_model.load_weights(weight_path)

        self.keras_model.compile(loss={'CTC_Loss': lambda y_true, FramePred: FramePred}, optimizer="Adadelta")
{% endhighlight %}

As you can see if you have already read example code `image_ocr.py`, the core code is almost the same. Well, I have to confess that I waste a lot of time on using `K.ctc_cost` rather than batch cost. Because the glimpse of the example code gave me the impression that I have to pad the sequence. As CTC it self contains the combination of repeating predictions, it is quite unacceptable to interpret a single block of silence into a few continuous label sequence. But later I came to realize that in batch cost, the program managed to mask out padded sequence using `label_length`, so I returned to use batch cost.
But even though the code is almost the same as example, there are still few more traps if you start from the example code. Well, I will tell you about this later, let's get back onto the correct one now.

The motivation of the core code can be summed up as followings:
+ Giving framewise prediction and label sequence as well as their corresponding length to `ctc_batch_cost` function.
+ Incorporate the theano function into keras network by `Lambda` layer.
+ Compile the model using dummy loss.
In fact, I am not aware of how the loss is calculated, so I don't know why I have to set the loss in this way. Maybe I have to go through `compile` in keras. As stated by the original author, the actual loss calculation happens elsewhere and this loss is for completion.  
So we just need to pass the data as required.

Data preprocessing and training configuation:
{% highlight python %}
class BatchTrainingScheduler(object):
    """
    Schedule batch training of CTC network.
    Maybe more generally in the future
    """
    ### model is a keras model
    def __init__(self,model,name,X,Y,ph_Y,batch_size,epoch):
        """
        Take in the full dataset and desired epoch number, schedule a batch-based training and evaluation.
        This class is optimized for keras' 4D training data.
        i.e. X : (samples,channel,length,feature)
            Y : (samples,length,n_class)
        """
        self.X = X
        self.Y = Y
        self.ph_Y = ph_Y
        self.batch_size = batch_size
        self.batch_num = self.X.shape[0] // batch_size
        self.tot_epoch = epoch
        self.epoch = 0
        self.Net = model
        self.index = 0
        self.name = name

    ### Train for one epoch
    def train_step(self):
        bi = 0
        ei = 0
        # i is batch index
        for i in range(self.batch_num):
            bi = i * self.batch_size
            ei = bi + self.batch_size
            if ei > self.X.shape[0]:
                ei = self.X.shape[0]
            train_x = self.X[bi:ei,:,:,:]
            train_y = np.zeros((self.batch_size, self.Net.maxil))
            len_y = np.array([len(self.Y[i+bi]) for i in range(ei-bi)])
            len_pred = np.array([self.Net.input_len for i in range(self.batch_size)])
            for j in range(self.batch_size):
                train_y[j,:len_y[j] ] = self.Y[bi+j][:].astype("int32")

            #print(len_pred)
            #print(len_y)
            #print(train_x.shape)
            #print(train_y.shape)

            res = self.Net.keras_model.train_on_batch({"CTC_Label":train_y,"Framewise_Input":train_x,"label_length":len_y,"input_length":len_pred},np.zeros_like(train_y))
            print(res)

    def execute(self):
        print("Training for %d epoches" % self.tot_epoch)
        for i in range(self.tot_epoch):
            print("Epoch %d:" % i)
            self.train_step()
            self.save_weights(i)
            print("Shuffling data...")
            self.X, self.ph_Y, self.Y = shuffle_data(self.X,self.ph_Y,self.Y)
{% endhighlight %}

And here comes the most tricky step, which trapped me for nearly one week:
## export THEANO_FLAGS=optimizer=fast_compile
Well,it may not be a indispensible one. After I experimented on more machines, I found the need for this flag vary with machine and even model scale. For example, constructing a 2-layer bi-directional LSTM will require this flag in one machine while 1-layer will not. And in my personal laptop (which is CPU-only) all the model require settting this flag.

It is quite an unbelievable setting. Maybe I have not really cleared the bugs. Setting the `optimizer` configuration involved with Theano graph optimization. This configuration is `fast_run` by default. Setting this flag to fast_compile or None both works. If you try to compile the model normally, you will face the following bugs:
+ The model cannot compile in stacking the second layer of LSTM.
+ (If you reduce the model to one layer of BiLSTM) Theano will report `scan` using regative index (perhaps -37)
Unbelievable. You may imagine the how a Theano and Keras beginner feels when he tried for tens of times and confirmed this issue. Please help me if you have any idea. I'd better write my code for another time before I assume it is a bug of Theano :) .
## But, it works.
### And I would be grateful if you can tell me why.

Any way, using the following configuration, it is expected to setup CTC training.
But there is still one problem: Training tend to produce `nan` after a few iteration. And it is quite stochastic. If you use SGD with learning rate`0.001`, you will get `nan` after about 15 iterations. If you use `0.0001` SGD, `nan` will appear  in 100 iteration. If you use `Adadelta` or `Adam`, loss will quickly go down to about 800 but then become `nan`.
Currently, if you are training a model similar to my model shown above, you should set learning rate to `10-6` (for about 64 convolution filters). If you use single layer BLSTM, then `10-5` is ok.

To sum up, though keras provide a convenient solution for regular network configuration, we have to use more fundamental toolkits such as Theano if we need to train unusual networks. For the next few days, I will dive into using `raw` theano to build a CTC network.


If you use the code or this post elsewhere, please show the credit :)
