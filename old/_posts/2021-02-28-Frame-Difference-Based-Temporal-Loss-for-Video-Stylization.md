---
layout: post
title:  "[Demo] Frame-Difference Temporal Loss for Video Stylization"
date:   2021-02-28 09:00:00 +0800
categories: jekyll update
---

Video stylization means to stylize videos into having similar styles like artworks. For example, stylize a video of the subway into Van Gogh's Starry Night. Image stylization methods can be applied to individual frames of the original video to get a stylization of the video. However, the results will often contain flickering artifacts, which are of high-frequency and make the video look unpleasant. This type of artifact is called short-term inconsistency by literature. Another inconsistency is long-term, which means that the appearance of one object changes after occlusion.

In order to address the temporal inconsistency problems, researchers basically follow two paths: (1) Build a temporal consistency loss function based on optic flow, and finetune the stylization network using this loss.  (2) Estimate optic flow between the current frame and the previous frame and directly wrap content from the previous stylized frame. However, optic flow is challenging to estimate, not only error-prone but also time-consuming. Therefore, we want to find an alternative for the optic flow-based methods.

Our method follows the first category of methods, which is to build a new temporal loss. Our core idea is to use frame difference. Frame difference means the difference between the current frame and the previous frame. The frame difference of the original video depicts the change of all pixels along the time axis, i.e., the changing dynamics is encoded in frame difference. On the other hand, the frame difference of stylized outputs is reflects another changing dynamics of pixels. When high frequency flickering artifacts appear, the dynamics of pixels of stylized outputs is significantly different than the corresponding dynamics of inputs. Therefore, our frame difference-based temporal loss is formed by matching the frame difference of stylized outputs to the original videos.

I build a simple user study `game` in my blog to help demonstrate our method.

- [Evaluation of video stability](https://atlantixjj.github.io/jekyll/update/2018/11/13/video-stability.html) 

- [Evaluation of frame quality](https://atlantixjj.github.io/jekyll/update/2018/11/12/frame-quality.html)