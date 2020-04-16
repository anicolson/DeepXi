
<!---

export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

-->

Deep Xi has been updated to TensorFlow 2!
====

Deep Xi: *A Deep Learning Approach to *A Priori* SNR Estimation.*
====

Deep Xi is implemented in TensorFlow 2 and is used for speech enhancement, noise estimation, for mask estimation, and as a front-end for robust ASR.
----
[Deep Xi](https://doi.org/10.1016/j.specom.2019.06.002) (where the Greek letter 'xi' or ξ is pronounced  /zaɪ/) is a deep learning approach to *a priori* SNR estimation that was proposed in [[1]](https://doi.org/10.1016/j.specom.2019.06.002) and is implemented in [TensorFlow 2](https://www.tensorflow.org/). Some of its use cases include:


* It can be used by minimum mean-square error (MMSE) approaches to **speech enhancement** like the MMSE short-time spectral amplitude (MMSE-STSA) estimator.
* It can be used by minimum mean-square error (MMSE) approaches to **noise estimation**, as in *DeepMMSE* [[2]](https://ieeexplore.ieee.org/document/9066933).
* Estimate the ideal binary mask **(IBM)** for missing feature approaches or the ideal ratio mask **(IRM)**.
* A **front-end for robust ASR**, as shown in **Figure 1**.

|![](./fig_front-end.png "Deep Xi as a front-end for robust ASR.")|
|----|
| <p align="center"> <b>Figure 1:</b> Deep Xi used as a front-end for robust ASR. The back-end (Deep Speech) is available <a href="https://github.com/mozilla/DeepSpeech">here</a>. The noisy speech magnitude spectrogram, as shown in <b>(a)</b>, is a mixture of clean speech with <i>voice babble</i> noise at an SNR level of -5 dB, and is the input to Deep Xi. Deep Xi estimates the <i>a priori</i> SNR, as shown in <b>(b)</b>. The <i>a priori</i> SNR estimate is used to compute an MMSE approach gain function, which is multiplied elementwise with the noisy speech magnitude spectrum to produce the clean speech magnitude spectrum estimate, as shown in <b>(c)</b>. <a href="https://github.com/anicolson/matlab_feat">MFCCs</a> are computed from the estimated clean speech magnitude spectrogram, producing the estimated clean speech cepstrogram, as shown in <b>(d)</b>. The back-end system, Deep Speech, computes the hypothesis transcript, from the estimated clean speech cepstrogram, as shown in <b>(e)</b>. </p> |


How does Deep Xi work?
----
A training example is shown in **Figure 2**. A deep neural network (DNN) within the Deep Xi framework is fed the **noisy-speech short-time magnitude spectrum** as input. The training target of th DNN is a mapped version of the instantaneous *a priori* SNR (i.e. **mapped *a priori* SNR**). The instantaneous *a priori* SNR is mapped to the interval `[0,1]` in order to improve the rate of convergence of the used stochastic gradient descent algorith. The map is the cumulative distribution function (CDF) of the instantaneous *a priori* SNR, as given by Equation (13) in [[1]](https://doi.org/10.1016/j.specom.2019.06.002). The statistics for the CDF are computed over a sample of the training set. An example of the mean and standard deviation of the sample for for each frequency bin is shown in **Figure 3**. The training examples in each mini-batch are padded to the longest sequence length in the mini-batch. The **sequence mask** is used by TensorFlow to ensure that the DNN is not trained on the padding. During inference, the *a priori* SNR estimate is computed from the mapped *a priori* SNR using the sample statistics and Equation (12) from [[2]](https://ieeexplore.ieee.org/document/9066933).

|![](./fig_training_example.png "Deep Xi training example.")|
|----|
| <p align="center"> <b>Figure 2:</b> <a>A training example for Deep Xi. Generated using `eval_example.m`.</a> </p> |

 |![](./fig_xi_dist.png "Normal distribution of the instantaneous *a priori* SNR in dB for each frequency bin.")|
|----|
| <p align="center"> <b>Figure 3:</b> <a>The normal distribution for each frequency bin is computed from the mean and standard deviation of the instantaneous *a priori* SNR (dB) over the sample. Generated using `eval_stats.m`</a> </p> |

Which audio do I use with Deep Xi?
----
Deep Xi operates on mono/single-channel audio (not stereo/dual-channel audio). Single-channel audio is commonly due to most cell phones using a single microphone. The available trained models operate on a sampling frequency of `f_s=16000`Hz, which is currently the standard sampling frequency used in the speech enhancement community. The sampling frequency can be changed in `run.sh`. Deep Xi can be trained using a higher sampling frequency (e.g. `f_s=44100`Hz), but this is unnecessary as human speech rarely exceeds 8 kHz (the Nyquist frequency of `f_s=16000`Hz is 8 kHz). The available trained models operate on a window duration and shift of `T_d=32`ms and `T_s=16`ms. To train a model on a different window duration and shift, `T_d` and `T_s` can be changed in `run.sh`. Currently, Deep Xi supports `.wav`, `.mp3`, and `.flac` audio codecs. The audio codec and bit rate does not affect the performance of Deep Xi.

Where can I get a dataset for Deep Xi?
----
Open-source training and testing sets are available for Deep Xi on IEEE *DataPort*:

Deep Xi Training Set: [http://dx.doi.org/10.21227/3adt-pb04](http://dx.doi.org/10.21227/3adt-pb04).

Deep Xi Test Set: [http://dx.doi.org/10.21227/h3xh-tm88](http://dx.doi.org/10.21227/h3xh-tm88).

Test set from the original [Deep Xi paper](https://doi.org/10.1016/j.specom.2019.06.002): [http://dx.doi.org/10.21227/0ppr-yy46](http://dx.doi.org/10.21227/0ppr-yy46).

The MATLAB scripts used to generate these sets can be found in [`set`](https://github.com/anicolson/DeepXi/tree/master/set).

Current networks
-----
The following networks are **causal**. This is facilitated by using unidirectional recurrent cells, causal convlutional kernels, and layer normalisation that does not consider future time-frames (`frame-wise layer normalisation`).

**Deep Xi-TCN**

Deep Xi utilising a temporal convolutional network (TCN) was proposed in [[2]](https://ieeexplore.ieee.org/document/9066933). It uses bottleneck residual blocks and a cyclic dilation rate. The network comprises of approximately 2 million parameters and has a contextual field of approximately 8 seconds. The configuration of `tcn-1a` is as follows:
```
d_model=256
n_blocks=40
d_f=64
k=3
max_d_rate=16
test_epoch=
mbatch_size=8
```
An example of Deep Xi-TCN is shown in **Figure 4**.

**Deep Xi-ResLSTM**

Deep Xi utilising a residual long short-term memory (ResLSTM) network was proposed in [[1]](https://doi.org/10.1016/j.specom.2019.06.002). It uses residual blocks containing a single LSTM cell. The network comprises of approximately 10 million parameters. The configuration of `reslstm-1a` is as follows:
```
d_model=512  
n_blocks=5   
test_epoch=  
mbatch_size=8   
```

Availability
-----
A trained model for version **tcn-1a** is available in the [`model`](https://github.com/anicolson/DeepXi/tree/master/model) directory. It is trained using the [Deep Xi Training Set](https://ieee-dataport.org/open-access/deep-xi-training-set).

|![](./fig_Deep-Xi-ResNet.png "Deep Xi-TCN a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 4:</b> <a> <b>(left)</b> Deep Xi-TCN with <i>B</i> bottlekneck blocks. Each block has a bottlekneck size of <i>d_f</i>, and an output size of <i>d_model</i>. The middle convolutional unit has a kernel size of <i>k</i> and a dilation rate of <i>d</i>. The input to the TCN is the noisy speech magnitude spectrum for frame <i>l</i>.  The output is the corresponding mapped <i>a priori</i> SNR estimate for each component of the noisy speech magnitude spectrum. <b>(right)</b> An example of Deep Xi-TCN with <i>B=6</i>, a kernel size of <i>k=3</i>, and a maximum dilation rate of <i>4</i>. The dilation rate increases with the block index, <i>b</i>, by a power of 2 and is cycled if the maximum dilation rate is exceeded.</a></p> |

<!--
Trained models for **c2.7a** and **c1.13a** can be found in the *./model* directory. The trained model for **n1.9a** is to large to be stored on github. A model for **n1.9a** can be downloaded from [here](https://www.dropbox.com/s/wkhymfmx4qmqvg7/n1.5a.zip?dl=0).
-->

Results
-----
TO BE UPDATED

Objective scores obtained on the test set described [here](http://ssw9.talp.cat/papers/ssw9_PS2-4_Valentini-Botinhao.pdf). As in previous works, the objective scores are averaged over all tested conditions. **CSIG**, **CBAK**, and **COVL** are mean opinion score (MOS) predictors of the signal distortion, background-noise intrusiveness, and overall signal quality, respectively. **PESQ** is the perceptual evaluation of speech quality measure. **STOI** is the short-time objective intelligibility measure (in \%). The highest scores attained for each measure are indicated in boldface.

| Method                     | Causal | CSIG | CBAK | COVL | PESQ | STOI      |
|----------------------------|--------|------|------|------|------|-----------|
| Noisy speech               | --     | 3.35 | 2.44 | 2.63 | 1.97 | 92 (91.5) |
| [Wiener](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=543199) | Yes    | 3.23 | 2.68 | 2.67 | 2.22 | --        |
| [SEGAN](https://arxiv.org/pdf/1703.09452.pdf)                      | No     | 3.48 | 2.94 | 2.80 | 2.16 | **93**        |
| [WaveNet](https://arxiv.org/pdf/1706.07162.pdf)                    | No     | 3.62 | 3.23 | 2.98 | --   | --        |
| [MMSE-GAN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462068)                 | No     | 3.80 | 3.12 | 3.14 | 2.53 | **93**        |
| [Deep Feature Loss](https://arxiv.org/pdf/1806.10522.pdf)          | Yes    | 3.86 | **3.33** | 3.22 | --   | --        |
| [Metric-GAN](https://arxiv.org/pdf/1905.04874.pdf)                 | No     | 3.99 | 3.18 | 3.42 | **2.86** | --        |
| **Deep Xi (ResNet 3e, MMSE-LSA)** | Yes    | **4.12** | **3.33** | **3.48** | 2.82 | **93 (93.3)** |

Installation
-----

Prerequisites for GPU usage:

* [CUDA 10.01](https://developer.nvidia.com/cuda-downloads)
* [cuDNN (>= 7.6)](https://developer.nvidia.com/cudnn)

To install:

1. `git clone https://github.com/anicolson/DeepXi.git`
2. `virtualenv --system-site-packages -p python3 ~/venv/DeepXi`
3. `source ~/venv/DeepXi/bin/activate`
4. `cd DeepXi`
5. `pip install -r requirements.txt`

How to Use the Deep Xi
-----

Use [`run.sh`](https://github.com/anicolson/DeepXi/blob/master/run.sh) to configure and run Deep Xi.

**Inference:**
To perform inference and save the outputs, use the following:
```
./run.sh NETWORK="TCN" INFER=1 GAIN="mmse-lsa"
```
Please look in [`thoth/args.py`](https://github.com/anicolson/DeepXi/blob/master/deepxi/args.py) for available gain functions and [`run.sh`](https://github.com/anicolson/DeepXi/blob/master/run.sh) for further options.

**Testing:**
To perform testing and get objective scores, use the following:
```
./run.sh NETWORK="TCN" TEST=1 GAIN="mmse-lsa"
```
Please look in [`log/results`](https://github.com/anicolson/DeepXi/blob/master/log/results) for the results.

**Training:**
```
./run.sh NETWORK="TCN" TRAIN=1 GAIN="mmse-lsa"
```
Ensure to delete the data directory before training. This will allow training lists and statistics for your training set to be saved and used. **To retrain from a certain epoch, set `--resume_epoch` in [`run.sh`](https://github.com/anicolson/DeepXi/blob/master/run.sh) to the desired epoch**.

Citation guide
-----
Please cite the following depending on what you are using:
* If using Deep Xi-ResLSTM, please cite [1].
* If using Deep Xi-TCN, please cite [1] and [2].
* If using DeepMMSE, please cite [2].

[1] [A. Nicolson, K. K. Paliwal, Deep learning for minimum mean-square error approaches to speech enhancement, Speech Communication 111 (2019) 44 - 55, https://doi.org/10.1016/j.specom.2019.06.002.](https://doi.org/10.1016/j.specom.2019.06.002)

[2] [Q. Zhang, A. M. Nicolson, M. Wang, K. Paliwal and C. Wang, "DeepMMSE: A Deep Learning Approach to MMSE-based Noise Power Spectral Density Estimation," in IEEE/ACM Transactions on Audio, Speech, and Language Processing.](https://ieeexplore.ieee.org/document/9066933)
