# Detectron In a Day

By Atlantix, 2018/5/29.

## Environment Configuration

MSCOCO dataset is already prepared in the server.

### COCO API

Clone from its github page and build as requested.

Obstacle 1: `X display` not set properly.

Solution: Reinstall `xorg`.

Obstacle 2: `undefined symbol: PyFPE_jbuf`. A rare issue, when you compile it before you change the python library this will happen. For example, switching from anaconda3 to anaconda2. The solution is to clean all the build and make again.

### Caffe2 Installation (From Source)

Do as the official website suggest:

https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile

But there are many problems that force me to switch to anaconda.

Part of the issues are the followings:

1. cmake failure: `string does not recognize sub-command APPEND`. Probably caused by low version of CMake. To deal with it, you need to install the latest CMake from source.

2. make failure。`/usr/bin/ld: cannot find -lnvrtc`. After examing its code, I found `sudo` to be part of the failure. Use user permission to build the source will eliminate this failure.

3. opencv linking failure。`/usr/local/lib/libopencv_core.so.3.4: undefined reference to dgeqrf_`。It is probably the problem with opencv pre-built binaries. To solve it, you need to build opencv from source.

Well, I stop at the 3rd step. It is too annoying.

### Caffe2 Installation (By Anaconda)

It seems that `detectron` only support python2. So unfortunate for Anaconda3 users.

There is no problem installing anaconda2, and install Caffe2 is one line:

```
conda install -c caffe2 caffe2-cuda8.0-cudnn7
```

I notice that the official instruction suggest to add `gcc4.8` subfix if the GCC version is lower than 5. But if you do that, the package cannot be found. And Continue without subfix seems to be fine. Anyway.

But later, there is other issues with CUDA library. Lots of version problem:

1. CuDNN need to be 7.

2. NCCL need to be 2.

Download from NVIDIA and install to local directory solve the problem. I suppose you do not have sudo permission, in this case fake a `usr/local/` in your local directory will help.

Then I verified the installation. Successful!

### Compile Detectron

The instruction is very simple:

```
pip install -r ./requirements.txt --user
make
python2 detectron/tests/test_spatial_narrow_as_op.py
```

However `pip install -r ./requirements.txt --user` cause serious problem in my case. It seems that pip fails to recognize conda, and the libraries are installed twice. As a result, the whole python environment breaks. It is more convenient reinstalling anaconda than repairing it.

After this, I continue without error.

### Train MaskRCNN with Detectron

Inference:

```
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir ./log/detectron-visualizations \
    --image-ext jpg \
    --wts ../model/model_final.pkl \
    demo
```

Train (End to End):

```
python2 tools/train_net.py \
    --cfg configs/my/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR ./log/detectron-mye2e_2
```

Modifications needed are:

1. Reduce NUM_GPU to suit your machine.

2. Add dataset in `dataset_catalog.py` if your dataset is not at the particular position.

3. Modify dataset name too.

4. Learning rate schedule modification. If you are not using 8 GPUs, then you should do a linear learning rate schedule stretch according to `getting_started` section.

### Experiment Log

#### Expr 1

```
python2 tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/my/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR ./log/detectron-mye2e_2
```

In this experiment, I forgot to stretch the learning rate schedule, and it failed with learning, resulting in `Negative Area` fault.

#### Expr 2

All the settings are done properly. But the training takes totally 4 days on 2 GPUs.