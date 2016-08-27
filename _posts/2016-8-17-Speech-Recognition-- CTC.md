---
layout: post
title:  "Speech Recognition: Connectionist Temporal Classification (Unfinished)"
date:   2016-08-17 17:03:23 +0800
categories: jekyll update
---
## Man, don't be frightened by CTC!

If you are just diving into Speech Recognition and scan through recent awesome papers, you will find yourself facing a disgusting word `CTC`. You know that it stands for `Connectionist Temporal Classification` but this phrase disgusts you again. The papers just state that ' run forward and backward and sum over' which means nothing to you and you even don't know what this thing is for. ( If you are familiar with HMM, you may understand `CTC` easier.  This article is aimed at people with zero fundamentals. )

Well, before you stuck into Graves' origin paper ![`Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks`]({{site.baseurl}}/assets/icml2006_GravesFGS06.pdf) , I would like to give you a brief picture of CTC, so you won't result in wasting a lot of time like me. I suppose you are not beginners of Neural Networks in the following explanation.

Assume you have a RNN classifier which takes an audio file as input. To simplify, the audio file is transformed into spectrum, consisting of a series of framewise feature vectors (which are FFT coefficients). Your RNN then takes in a frame and it is supposed to output a label at a timestep, which is one of the 61 phonemes (plus a silence notation). If you have already trained the RNN, you can expect getting the right phoneme at the right time, i.e. , inside the phoneme segmentation label. And you may train the network using the segmentation label, too. Everything goes fine, isn't it?

So now you may wonder how can one train a framewise classifier with big data? Since phoneme-labeled data is rather rare and effort-taking. Is there any technique that allow end-to-end training of framewise RNN without precisely segmented data? This problem sounds like producing a sequence-level prediction from framewise predictions. It is actually an ancient issue in HMM: Given an observation sequence and HMM model, find the most probable hidden state sequence. We have network output as observation (and a language model if needed) and every frame's final decision as hidden state and a transition matrix as HMM parameter. This part is just a forward recursion and quite easy to understand. Just google `HMM` or `viterbi`.

But it is actually quite different one. Well, you are actually modifying the network objective function! In order to train RNN end-to-end with sequence-level labelings, it is essential to derive a differentiable loss function that can describe sequence-level error signal. This sounds much harder. (And you may feel even harder to implement its symbolic computation graph using Theano).
