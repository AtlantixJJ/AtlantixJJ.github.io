---
layout: post
title:  "Neural Painter: A smart image editor by simple line drawings"
date:   2018-09-01 08:00:00 +0800
categories: jekyll update
---

The online demo is available at [neuralpainter.club:2333](neuralpainter.club:2333). (Update: currently this server is down for maintainence, online demo is temporarily unavailable.) And you can watch a short demo video at the bottom of this page.

Visual language is an important part of human communication, however most of people are not skilled at expressing themselves visually. This project aims to bridge the expertise gap of drawing by Interactive Generative Adversarial Network (iGAN). iGAN is proposed by Junyan Zhu in 2015, in their paper a fundamental optimization based image edit framework is proposed and since widely adapted, e.g. Neural Photo Editor in 2016. We base our image editor on this framework while build a more power Generator and Discriminator architecture.

To be honest, the current project is not featured in academic novelty, though I am actively exploring novel method for line-drawing based image editing. Instead, this project is featured at excellent performance in practice, and a well-implemented edit system, including a nice looking UI, a backend optimized for midium scale online usage.

In addition, the model can only support edition for specific dataset object, like shoes, handbags, churches, outdoor scenes, human faces... And we train our model to fit on anime character face dataset! For all those object mentioned, we have trained and validate these models, but they are not online because of compuation budget issue.

<iframe width="960" height="540" src="https://www.youtube.com/embed/Il596wgjUc8" frameborder="0" allowfullscreen></iframe>