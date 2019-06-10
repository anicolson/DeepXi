[//]: # export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
[//]: # export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

Deep Xi: A Deep Learning Approach to *A Priori* SNR Estimation. Used for Speech Enhancement and Robust ASR.
====

Deep Xi (where the Greek letter 'xi' or ξ is pronounced  /zaɪ/) is a deep learning approach to *a priori* SNR estimation that was proposed in [1]. It can be used by minimum mean-square error (MMSE) approaches to **speech enhancement** like the MMSE short-time spectral amplitude (MMSE-STSA) estimator, the MMSE log-spectral amplitude (MMSE-LSA) estimator, and the Wiener filter (WF) approach. It can also be used to estimate the ideal ratio mask (IRM) and the ideal binary mask (IBM). Deep Xi can be used as a **front-end for robust ASR**, as shown in **Figure 1**. Deep Xi is implemented in [TensorFlow](https://www.tensorflow.org/).

|![](./fig_front-end.png "Deep Xi as a front-end for robust ASR.")|
|----|
| <p align="center"> <b>Figure 1:</b> Deep Xi used as a front-end for robust ASR. The back-end (Deep Speech) is available <a href="https://github.com/mozilla/DeepSpeech">here</a>. The noisy speech magnitude spectrogram, as shown in <b>(a)</b>, is a mixture of clean speech with <i>voice babble</i> noise at an SNR level of -5 dB, and is the input to Deep Xi. Deep Xi estimates the <i>a priori</i> SNR, as shown in <b>(b)</b>. The <i>a priori</i> SNR estimate is used to compute an MMSE approach gain function, which is multiplied elementwise with the noisy speech magnitude spectrum to produce the clean speech magnitude spectrum estimate, as shown in <b>(c)</b>. <a href="https://github.com/anicolson/matlab_feat">MFCCs</a> are computed from the estimated clean speech magnitude spectrogram, producing the estimated clean speech cepstrogram, as shown in <b>(d)</b>. The back-end system, Deep Speech, computes the hypothesis transcript, from the estimated clean speech cepstrogram, as shown in <b>(e)</b>. </p> |

|![](./fig_tcn.gif "TCN a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 2:</b> <a> TCN </a> <i> a priori</i>  <a> SNR estimator.</a> </p> |

|![](./fig_reslstm.png "ResLSTM a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 3:</b> <a> ResLSTM </a> <i> a priori</i>  <a> SNR estimator.</a> </p> |

|![](./fig_resblstm.png "ResBLSTM a priori SNR estimator.")|
|----|
| <p align="center"> <b>Figure 4:</b> <a> ResBLSTM </a> <i> a priori</i>  <a> SNR estimator.</a> </p> |

Current Models
-----
The scripts for each of the following *a priori* SNR estimators can be found in the [*./ver* directory](https://github.com/anicolson/DeepXi/tree/master/ver):

* **c2.7a** is a TCN ([temporal convolutional network](https://arxiv.org/pdf/1803.01271.pdf)) that has 2 million parameters, as shown in **Figure 2**.
* **c1.13a** is a ResLSTM (residual long short-term memory network) with 10.8 million parameters, as shown in **Figure 3**.
* **n1.9a** is a ResBLSTM (residual bidirectional long short-term memory network) with 21.3 million parameters, as shown in **Figure 4**.

**'c'** and **'n'** indicate if the system is *causal* or *non-causal*, respectively. **'c1'**, **'n1'**, and **'c2'** indicate if a ResLSTM, ResBLSTM, or a TCN network is used, respectively.

Model Availability
-----

* **c2.7a**  can be found in the [*./model* directory](https://github.com/anicolson/DeepXi/tree/master/model/c2.7a). 
* **c1.13a** can be downloaded from Dropbox [here](https://www.dropbox.com/s/tpe5ydj758jvic9/c1.13a.zip?dl=0), or the Nihao cloud service [here](https://app.nihaocloud.com/f/d5675749ba7342a09a61/?dl=1).
* **n1.9a** can be downloaded from Dropbox [here](https://www.dropbox.com/s/1o5d7pj2pinxitz/n1.9a.zip?dl=0), or the Nihao cloud service [here](https://app.nihaocloud.com/f/3739ce91061e4d619272/?dl=1).

Note: The trained models for **c1.13a** and **n1.9a** are too large to be stored on github.

<!--
Trained models for **c2.7a** and **c1.13a** can be found in the *./model* directory. The trained model for **n1.9a** is to large to be stored on github. A model for **n1.9a** can be downloaded from [here](https://www.dropbox.com/s/wkhymfmx4qmqvg7/n1.5a.zip?dl=0). 
-->

Installation
-----

It is recommended to use a [virtual environment](http://virtualenvwrapper.readthedocs.io/en/latest/install.html) for installation.

Prerequisites:

* [TensorFlow ](https://www.tensorflow.org/) r1.11 (installed in a virtual environment). Will be updated to r2.0 in the near future.
* [Python3](https://docs.python-guide.org/starting/install3/linux/)
* [MATLAB](https://www.mathworks.com/products/matlab.html) (only required for .mat output files)

To install:

1. `git clone https://github.com/anicolson/DeepXi.git`
2. `pip install -r requirements.txt`

How to Use the Deep Xi Scripts
-----
Inference:

```
cd ver/c2/7/a
python3 deepxi.py --test 1 --out_type y --gain mmse-lsa --gpu 0
```
**y** for **--out_type** specifies enhanced speech .wav output. **mmse-lsa** specifies the used gain function (others include **mmse-stsa**, **wf**, **irm**, **ibm**, **srwf**, **cwf**).


Training:

```
cd ver/c2/7/a
python3 deepxi.py --train 1 --verbose 1 --gpu 0
```

Retraining:

```
cd ver/c2/7/a
python3 deepxi.py --train 1 --cont 1 --retrain_epoch 175 --verbose 1 --gpu 0
```

Other options can be found in the *deepxi.py* script.

Datasets Used For Training
-----
The .wav files used for training are single-channel, with a sampling frequency of 16 kHz.

The following speech datasets were used:
* The *train-clean-100* set from Librispeech corpus, which can be found [here](http://www.openslr.org/12/).
* The CSTR VCTK corpus, which can be found [here](https://datashare.is.ed.ac.uk/handle/10283/2651).
* The *si* and *sx* training sets from the TIMIT corpus, which can be found [here](https://catalog.ldc.upenn.edu/LDC93S1) (not open source).

The following noise datasets were used:
* The QUT-NOISE dataset, which can be found [here](https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/).
* The Nonspeech dataset, which can be found [here](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html). 
* The Environemental Background Noise dataset, which can be found [here](http://www.utdallas.edu/~nxk019000/VAD-dataset/).
* The noise set from the MUSAN corpus, which can be found [here](http://www.openslr.org/17/).
* Multiple packs from the FreeSound website, which can be found  [here](https://freesound.org/)

References
-----

[[1] A. Nicolson and K. K. Paliwal, "Deep Learning For Minimum Mean-Square Error Approaches to Speech Enhancement", Accepted to Speech Communication.](./xi_2018.pdf)
