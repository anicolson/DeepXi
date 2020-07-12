<!--

export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

-->

Deep Xi: *A Deep Learning Approach to *A Priori* SNR Estimation for speech enhancement.* 
====

Contents
----
  * [Introduction](#introduction-)
  * [How does Deep Xi work?](#how-does-deep-xi-work-)
  * [Current networks](#current-networks)
  * [Available models](#available-models)
  * [Results for Deep Xi Test Set](#results-for-deep-xi-test-set)
  * [Results for the DEMAND -- Voice Bank test set](#results-for-the-demand----voice-bank-test-set)
  * [DeepMMSE](#deepmmse)
  * [Installation](#installation)
  * [How to use Deep Xi](#how-to-use-deep-xi)
  * [Current issues and potential areas of improvement](#current-issues-and-potential-areas-of-improvement)
  * [Where can I get a dataset for Deep Xi?](#where-can-i-get-a-dataset-for-deep-xi-)
  * [Which audio do I use with Deep Xi?](#which-audio-do-i-use-with-deep-xi-)
  * [Naming convention in the `set/` directory](#naming-convention-in-the--set---directory)
  * [Citation guide](#citation-guide)


Introduction
----

**Deep Xi is implemented in TensorFlow 2 and is used for speech enhancement, noise estimation, for mask estimation, and as a front-end for robust ASR.** [Deep Xi](https://doi.org/10.1016/j.specom.2019.06.002) (where the Greek letter 'xi' or ξ is pronounced  /zaɪ/) is a deep learning approach to *a priori* SNR estimation that was proposed in [[1]](https://doi.org/10.1016/j.specom.2019.06.002) and is implemented in [TensorFlow 2](https://www.tensorflow.org/). Some of its use cases include:
* It can be used by minimum mean-square error (MMSE) approaches to **speech enhancement** like the MMSE short-time spectral amplitude (MMSE-STSA) estimator.
* It can be used by MMSE-based **noise PSD estimators**, as in *DeepMMSE* [[2]](https://ieeexplore.ieee.org/document/9066933).
* Estimate the ideal binary mask **(IBM)** for missing feature approaches or the ideal ratio mask **(IRM)**.
* A **front-end for robust ASR**, as shown in **Figure 1**.

|![](./docs/fig_front-end.png "Deep Xi as a front-end for robust ASR.")|
|----|
| <p align="center"> <b>Figure 1:</b> Deep Xi used as a front-end for robust ASR. The back-end (Deep Speech) is available <a href="https://github.com/mozilla/DeepSpeech">here</a>. The noisy speech magnitude spectrogram, as shown in <b>(a)</b>, is a mixture of clean speech with <i>voice babble</i> noise at an SNR level of -5 dB, and is the input to Deep Xi. Deep Xi estimates the <i>a priori</i> SNR, as shown in <b>(b)</b>. The <i>a priori</i> SNR estimate is used to compute an MMSE approach gain function, which is multiplied elementwise with the noisy speech magnitude spectrum to produce the clean speech magnitude spectrum estimate, as shown in <b>(c)</b>. <a href="https://github.com/anicolson/matlab_feat">MFCCs</a> are computed from the estimated clean speech magnitude spectrogram, producing the estimated clean speech cepstrogram, as shown in <b>(d)</b>. The back-end system, Deep Speech, computes the hypothesis transcript, from the estimated clean speech cepstrogram, as shown in <b>(e)</b>. </p> |

How does Deep Xi work?
----
A training example is shown in **Figure 2**. A deep neural network (DNN) within the Deep Xi framework is fed the **noisy-speech short-time magnitude spectrum** as input. The training target of the DNN is a mapped version of the instantaneous *a priori* SNR (i.e. **mapped *a priori* SNR**). The instantaneous *a priori* SNR is mapped to the interval `[0,1]` to improve the rate of convergence of the used stochastic gradient descent algorithm. The map is the cumulative distribution function (CDF) of the instantaneous *a priori* SNR, as given by Equation (13) in [[1]](https://doi.org/10.1016/j.specom.2019.06.002). The statistics for the CDF are computed over a sample of the training set. An example of the mean and standard deviation of the sample for each frequency bin is shown in **Figure 3**. The training examples in each mini-batch are padded to the longest sequence length in the mini-batch. The **sequence mask** is used by TensorFlow to ensure that the DNN is not trained on the padding. During inference, the *a priori* SNR estimate is computed from the mapped *a priori* SNR using the sample statistics and Equation (12) from [[2]](https://ieeexplore.ieee.org/document/9066933).

|![](./docs/fig_training_example.png "Deep Xi training example.")|
|----|
| <p align="center"> <b>Figure 2:</b> <a>A training example for Deep Xi. Generated using `eval_example.m`.</a> </p> |

 |![](./docs/fig_xi_dist.png "Normal distribution of the instantaneous *a priori* SNR in dB for each frequency bin.")|
|----|
| <p align="center"> <b>Figure 3:</b> <a>The normal distribution for each frequency bin is computed from the mean and standard deviation of the instantaneous *a priori* SNR (dB) over a sample of the training set. Generated using `eval_stats.m`</a> </p> |

Current networks
-----
Recurrent neural networks (RNNs) and temporal convolutional networks (TCNs), are available: <!-- and attention-based networks -->
<!--- * **MHANet**: Multi-head attention network. --->
* **ResLSTM**: Residual long short-term memory network [1].
* **ResNet**: Residual network [2].
* **RDLNet**: Residual-dense lattice network [3].

<!--- Deep Xi utilising the MHANet (**Deep Xi-MHANet**) was proposed in . --->

Deep Xi utilising a ResNet TCN (**Deep Xi-TCN**) was proposed in [[2]](https://ieeexplore.ieee.org/document/9066933). It uses bottleneck residual blocks and a cyclic dilation rate. The network comprises of approximately 2 million parameters and has a contextual field of approximately 8 seconds. An example of Deep Xi-ResNet is shown in **Figure 4**. Deep Xi utilising a ResLSTM network (**Deep Xi-ResLSTM**) was proposed in [[1]](https://doi.org/10.1016/j.specom.2019.06.002). Each of its residual blocks contain a single LSTM cell. The network comprises of approximately 10 million parameters.

|![](./docs/fig_Deep-Xi-ResNet.png "Deep Xi-ResNet a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 4:</b> <a> <b>(left)</b> Deep Xi-ResNet with <i>B</i> bottlekneck blocks. Each block has a bottlekneck size of <i>d_f</i>, and an output size of <i>d_model</i>. The middle convolutional unit has a kernel size of <i>k</i> and a dilation rate of <i>d</i>. The input to the ResNet is the noisy speech magnitude spectrum for frame <i>l</i>.  The output is the corresponding mapped <i>a priori</i> SNR estimate for each component of the noisy speech magnitude spectrum. <b>(right)</b> An example of Deep Xi-ResNet with <i>B=6</i>, a kernel size of <i>k=3</i>, and a maximum dilation rate of <i>4</i>. The dilation rate increases with the block index, <i>b</i>, by a power of 2 and is cycled if the maximum dilation rate is exceeded.</a></p> |

Available models
-----
There are multiple Deep Xi versions, comprising of different networks and restrictions. An example of the `ver` naming convention is `resnet-1.0c`. The network type is given at the start of `ver`. Versions with **c** are **causal**. Versions with **n** are **non-causal**.  The version iteration is also given, i.e. `1.0`. Here are the current versions:

**`resnet-1.1n` (available in the [`model`](https://github.com/anicolson/DeepXi/tree/master/model) directory)**

**`resnet-1.0n` (technically, this is not a TCN due to the use of non-causal dilated 1D kernels)**

**`resnet-1.0c` (available in the [`model`](https://github.com/anicolson/DeepXi/tree/master/model) directory)**

**`reslstm-1.0c` (there are issues with training this network currently)**

**Each available model is trained using the [Deep Xi Training Set](https://ieee-dataport.org/open-access/deep-xi-training-set). Please see [`run.sh`](https://github.com/anicolson/DeepXi/blob/master/run.sh) for more details about these networks.**


<!--
Trained models for **c2.7a** and **c1.13a** can be found in the *./model* directory. The trained model for **n1.9a** is to large to be stored on github. A model for **n1.9a** can be downloaded from [here](https://www.dropbox.com/s/wkhymfmx4qmqvg7/n1.5a.zip?dl=0).
-->

Results
-----

**Deep Xi Test Set**

Average objective scores obtained over the conditions in the [Deep Xi Test Set](http://dx.doi.org/10.21227/h3xh-tm88). SNR levels between -10 dB and 20 dB are considered only. **MOS-LQO** is the mean opinion score (MOS) objective listening quality score obtained using Wideband PESQ. **PESQ** is the perceptual evaluation of speech quality measure. **STOI** is the short-time objective intelligibility measure (in \%). **eSTOI** is extended STOI. **Results for each condition can be found in `log/results`**

| Method           | Gain      | Causal | MOS-LQO | PESQ | STOI | eSTOI |
|------------------|-----------|--------|---------|------|------|-------|
| Deep Xi-ResNet (1.0c) | MMSE-STSA | Yes    |   1.90|2.34|80.92|65.90|
| Deep Xi-ResNet (1.0c) | MMSE-LSA  | Yes    |   1.92|2.37|80.79|65.77|
| Deep Xi-ResNet (1.0c) | SRWF/IRM  | Yes    |   1.87|2.31|80.98|65.94|
| Deep Xi-ResNet (1.0c) | cWF       | Yes    |   1.92|2.34|81.11|65.79|
| Deep Xi-ResNet (1.0c) | WF        | Yes    |   1.75|2.21|78.30|63.96|
| Deep Xi-ResNet (1.0c) | IBM       | Yes    |   1.38|1.73|70.85|55.95|
| Deep Xi-ResNet (1.1n) | MMSE-LSA  | No     |   2.02|2.48|83.90|69.50|

**DEMAND -- Voice Bank test set**

Objective scores obtained on the DEMAND--Voicebank test set described [here](http://ssw9.talp.cat/papers/ssw9_PS2-4_Valentini-Botinhao.pdf). As in previous works, the objective scores are averaged over all tested conditions. **CSIG**, **CBAK**, and **COVL** are mean opinion score (MOS) predictors of the signal distortion, background-noise intrusiveness, and overall signal quality, respectively. **PESQ** is the perceptual evaluation of speech quality measure. **STOI** is the short-time objective intelligibility measure (in \%). The highest scores attained for each measure are indicated in boldface.

| Method                 |   Gain | Causal | CSIG | CBAK | COVL | PESQ | STOI      | SegSNR |
|-------------------------|---|--------|------|------|------|------|-----------|----|
| Noisy speech              | -- | --     | 3.35 | 2.44 | 2.63 | 1.97 | 92 (91.5) | -- |
| [Wiener](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=543199) |  | Yes    | 3.23 | 2.68 | 2.67 | 2.22 | --        | -- |
| [SEGAN](https://arxiv.org/pdf/1703.09452.pdf)                |   --   | No     | 3.48 | 2.94 | 2.80 | 2.16 | 93  |
| [WaveNet](https://arxiv.org/pdf/1706.07162.pdf)              | --     | No     | 3.62 | 3.23 | 2.98 | --   | --        |
| [MMSE-GAN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462068)    |    --         | No     | 3.80 | 3.12 | 3.14 | 2.53 | 93      | -- |
| [Deep Feature Loss](https://arxiv.org/pdf/1806.10522.pdf)  |  --      | Yes    | 3.86 | 3.33 | 3.22 | --   | --        | -- |
| [Metric-GAN](https://arxiv.org/pdf/1905.04874.pdf) |      --          | No     | 3.99 | 3.18 | 3.42 | 2.86 | --        | -- |
| **Deep Xi-ResNet (1.0c)** | MMSE-LSA | Yes    | 4.14 | 3.32 | 3.46 | 2.77 | 93 (93.2) | -- |
| **Deep Xi-ResNet (1.0n)** | MMSE-LSA | No    | 4.28 | 3.46 | 3.64 | 2.95 | 94 (93.6) | -- |
| **Deep Xi-ResNet (1.1c)** | MMSE-LSA | Yes    | 4.24 | 3.40 | 3.59 | 2.91 | 94 (93.5) | 8.4 |
| **Deep Xi-ResNet (1.1n)** | MMSE-LSA | No    | **4.32** | **3.52** | **3.68** | **3.01** | **94 (93.9)** | **9.4** |

DeepMMSE
----

Description coming soon...

Installation
-----

Prerequisites for GPU usage:

* [CUDA 10.1](https://developer.nvidia.com/cuda-downloads)
* [cuDNN (>= 7.6)](https://developer.nvidia.com/cudnn)

To install:

1. `git clone https://github.com/anicolson/DeepXi.git`
2. `virtualenv --system-site-packages -p python3 ~/venv/DeepXi`
3. `source ~/venv/DeepXi/bin/activate`
4. `cd DeepXi`
5. `pip install -r requirements.txt`

How to use Deep Xi
-----

Use [`run.sh`](https://github.com/anicolson/DeepXi/blob/master/run.sh) to configure and run Deep Xi.

**Inference:**
To perform inference and save the outputs, use the following:
```
./run.sh VER="resnet-1.1n" INFER=1 GAIN="mmse-lsa"
```
Please look in [`thoth/args.py`](https://github.com/anicolson/DeepXi/blob/master/deepxi/args.py) for available gain functions and [`run.sh`](https://github.com/anicolson/DeepXi/blob/master/run.sh) for further options.

**Testing:**
To perform testing and get objective scores, use the following:
```
./run.sh VER="resnet-1.1n" TEST=1 GAIN="mmse-lsa"
```
Please look in [`log/results`](https://github.com/anicolson/DeepXi/blob/master/log/results) for the results.

**Training:**
```
./run.sh VER="resnet-1.1n" TRAIN=1 GAIN="mmse-lsa"
```
Ensure to delete the data directory before training. This will allow training lists and statistics for your training set to be saved and used. **To retrain from a certain epoch, set `--resume_epoch` in [`run.sh`](https://github.com/anicolson/DeepXi/blob/master/run.sh) to the desired epoch**.

Current issues and potential areas of improvement
-----

If you would like to contribute to Deep Xi, please investigate the following and compare it to current models:

* Currently, the ResLSTM network is not performing as well as expected (when compared to TensorFlow 1.x performance).


Where can I get a dataset for Deep Xi?
----
Open-source training and testing sets are available for Deep Xi on IEEE *DataPort*:

Deep Xi Training Set: [http://dx.doi.org/10.21227/3adt-pb04](http://dx.doi.org/10.21227/3adt-pb04).

Deep Xi Test Set: [http://dx.doi.org/10.21227/h3xh-tm88](http://dx.doi.org/10.21227/h3xh-tm88).

Test set from the original [Deep Xi paper](https://doi.org/10.1016/j.specom.2019.06.002): [http://dx.doi.org/10.21227/0ppr-yy46](http://dx.doi.org/10.21227/0ppr-yy46).

The MATLAB scripts used to generate these sets can be found in [`set`](https://github.com/anicolson/DeepXi/tree/master/set).

Which audio do I use with Deep Xi?
----
Deep Xi operates on mono/single-channel audio (not stereo/dual-channel audio). Single-channel audio is used due to most cell phones using a single microphone. The available trained models operate on a sampling frequency of `f_s=16000`Hz, which is currently the standard sampling frequency used in the speech enhancement community. The sampling frequency can be changed in `run.sh`. Deep Xi can be trained using a higher sampling frequency (e.g. `f_s=44100`Hz), but this is unnecessary as human speech rarely exceeds 8 kHz (the Nyquist frequency of `f_s=16000`Hz is 8 kHz). The available trained models operate on a window duration and shift of `T_d=32`ms and `T_s=16`ms, respectively. To train a model on a different window duration and shift, `T_d` and `T_s` can be changed in `run.sh`. Currently, Deep Xi supports `.wav`, `.mp3`, and `.flac` audio codecs. The audio codec and bit rate does not affect the performance of Deep Xi.

Naming convention in the `set/` directory
-----

The following is already configured in the [Deep Xi Training Set](http://dx.doi.org/10.21227/3adt-pb04) and [Deep Xi Test Set](http://dx.doi.org/10.21227/h3xh-tm88).

**Training set**

The filenames of the waveforms in the `train_clean_speech` and `train_noise` directories are not restricted. There can be a different number of waveforms in each. The Deep Xi framework utilises each of the waveforms in `train_clean_speech` once during an epoch. For each `train_clean_speech` waveform of a mini-batch, the Deep Xi framework selects a random section of a randomely selected waveform from `train_noise` (that is at a length greater than or equal to the `train_clean_speech` waveform) and adds it to the `train_clean_speech` waveform at a randomly selected SNR level (the SNR level range can be set in `run.sh`).

**Validation set**

As the validation set must not change from epoch to epoch, a set of restrictions apply to the waveforms in `val_clean_speech` and `val_noise`. There must be the same amount of waveforms in `val_clean_speech` and `val_noise`. One waveform in `val_clean_speech` corresponds to only one waveform in `val_noise`, i.e. a clean speech and noise validation waveform pair. Each clean speech and noise validation waveform pair must have identical filenames and and an identical number of samples. Each clean speech and noise validation waveform pair must have the SNR level (dB) that they are to be mixed at placed at the end of their filenames. The convention used is `_XdB`, where `X` is replaced with the desired SNR level. E.g. `val_clean_speech/NAME_-5dB.wav` and `val_noise/NAME_-5dB.wav`. An example of the filenames for a clean speech and noise validation waveform pair is as follows: `val_clean_speech/198_19-198-0003_Machinery17_15dB.wav` and `val_noise/198_19-198-0003_Machinery17_15dB.wav`.


**Test set**

The filenames of the waveforms in the `test_noisy_speech` directory are not restricted. This is all that is required if you want inference outputs from Deep Xi, i.e. `./run.sh VER="ANY_NAME" INFER=1`. If you are obtaining objective scores by using `./run.sh VER="ANY_NAME" TEST=1`, then reference waveforms for the objective measures need to be placed in `test_clean_speech`. The waveforms in `test_clean_speech` and `test_noisy_speech` that correspond to each other must have the same number of samples (i.e. the same sequence length). The filename of the waveform in `test_clean_speech` that corresponds to a waveform in `test_noisy_speech` must be contained in the corresponding test noisy speech waveforn filename. E.g. if the filename of a test noisy speech waveform is `test_noisy_speech/61-70968-0000_SIGNAL021_-5dB.wav`, then the filename of the corresponding test clean speech waveform must be contained in the filename of the test noisy speech waveform: `test_clean_speech/61-70968-0000.wav`. This is because a test clean speech waveform may be used as a reference for multiple waveforms in `test_noisy_speech` (e.g. `test_noisy_speech/61-70968-0000_SIGNAL021_0dB.wav`, `test_noisy_speech/61-70968-0000_SIGNAL021_5dB.wav`, and `test_noisy_speech/61-70968-0000_SIGNAL021_10dB.wav` are additional test noisy speech waveforms that the test clean speech waveform from the previous example is a reference for).

Citation guide
-----

Please cite the following depending on what you are using:
* The Deep Xi framework is proposed in [1].
* If using Deep Xi-ResLSTM, please cite [1].
* If using Deep Xi-ResNet, please cite [1] and [2].
* If using DeepMMSE, please cite [2].
* If using Deep Xi-RDLNet, please cite [1] and [3].

[1] [A. Nicolson, K. K. Paliwal, Deep learning for minimum mean-square error approaches to speech enhancement, Speech Communication 111 (2019) 44 - 55, https://doi.org/10.1016/j.specom.2019.06.002.](https://doi.org/10.1016/j.specom.2019.06.002)

[2] [Q. Zhang, A. M. Nicolson, M. Wang, K. Paliwal and C. Wang, "DeepMMSE: A Deep Learning Approach to MMSE-based Noise Power Spectral Density Estimation," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 1404-1415, 2020, doi: 10.1109/TASLP.2020.2987441.](https://ieeexplore.ieee.org/document/9066933)

[3] [Mohammad Nikzad, Aaron Nicolson, Yongsheng Gao, Jun Zhou, Kuldip K. Paliwal, and Fanhua Shang. "Deep residual-dense lattice network for speech enhancement". In AAAI Conference on Artificial Intelligence, pages 8552–8559, 2020](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-NikzadM.6844.pdf)
