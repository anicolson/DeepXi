Deep Xi: *A Priori* SNR Estimation Using Deep Learning
====

Deep Xi (where the Greek letter 'xi' or ξ is ponounced  /zaɪ/) is a deep learning method for *a priori* SNR estimation that was proposed in [1]. It can be used by minimum mean-square error (MMSE) approaches to speech enhancemnet like the MMSE short-time spectral amplitude (MMSE-STSA) estimator, the MMSE log-spectral amplitude (MMSE-LSA) estimator, and the Wiener filter (WF) approach. It can also be used to estimate the ideal ratio mask (IRM) and the ideal binary mask (IBM). Deep Xi can be used as a front-end for robust ASR, as shown in Figure 1. DeepXi is implemented in [TensorFlow](https://www.tensorflow.org/).

![](./fig_front-end.png "Deep Xi as a front-end for robust ASR.")
<p align="center">
  <b>Figure 1:</b> Deep Xi used as a front-end for robust ASR. Deep Speech is available <a href="https://github.com/mozilla/DeepSpeech">here</a>. The noisy speech magnitude spectrogram, as shown in (a), is a mixture of clean speech with <i>voice babble</i> noise at an SNR level of -5 dB, and is the input to Deep Xi. Deep Xi estimates the <i>a priori</i> SNR, as shown in (b). The <i>a priori</i> SNR estimate is used to compute an MMSE approach gain function, which is multiplied elementwise with the noisy speech magnitude spectrum to produce the clean speech magnitude spectrum estimate, as shown in (c). <a href="https://github.com/anicolson/matlab_feat">MFCCs</a> are computed from the estimated clean speech magnitude spectrogram, producing the estimated clean speech cepstrogram, as shown in (d). The back-end system, Deep Speech, computes the hypothesis transcript, from the estimated clean speech cepstrogram as shown in (e). 
</p>

![](./fig_resblstm.png "ResBLSTM a priori SNR estimator.")
<p align="center">
  <b>Figure 2:</b> <a> ResBLSTM </a> <i> a priori</i>  <a> SNR estimator.</a>
</p>

![](./fig_reslstm.png "ResLSTM a priori SNR estimator.")
<p align="center">
  <b>Figure 3:</b> <a> ResLSTM </a> <i> a priori</i>  <a> SNR estimator.</a>
</p>


Current Models
-----

The scripts for each of the following models can be found in the *./ver* directory:

* **c2.7a** is a TCN (temporal convolutional network) that has 2 million parameters.
* **c1.13a** is a ResLSTM (residual long short-term memory network) that has 10.8 million parameters.
* **n1.9a** is a ResBLSTM (residual bidirectional long short-term memory network) that has 21.3 million parameters.

Trained models for **c2.7a** and **c1.13a** can be found in the *./model* directory. The trained model for **n1.9a** is to large to be stored on github. A model for **n1.9a** can be downloaded from [here](https://www.dropbox.com/s/wkhymfmx4qmqvg7/n1.5a.zip?dl=0). 


Installation
-----

It is recommended to use a [virtual environment](http://virtualenvwrapper.readthedocs.io/en/latest/install.html) for installation.

Prerequisites:

* [TensorFlow](https://www.tensorflow.org/) (installed in a virtual environment)
* [Python3](https://docs.python-guide.org/starting/install3/linux/)
* [MATLAB](https://www.mathworks.com/products/matlab.html) (only required for .mat output files)

To install:

1. `git clone https://github.com/anicolson/DeepXi.git`
2. `pip install -r requirements.txt`

How to Perform Speech Enhancement
-----

Simply run the script (python3 deepxi.py). Run the script in the virtual environment that TensorFlow is installed in. The script has different inference options, and is also able to perform training if required.

Directory Description
-----

Directory | Description
--------| -----------  
lib | Functions for deepxi.py.
model | The directory for the model (the model must be [downloaded](https://www.dropbox.com/s/wkhymfmx4qmqvg7/n1.5a.zip?dl=0)).
noisy_speech | Noisy speech. Place noisy speech .wav files to be enhanced here.
output | DeepXi outputs, including the enhanced speech .wav output files.
stats | Statistics of a sample from the training set. The mean and standard deviation of the *a priori* SNR for the sample are used to compute the training target. 

References
-----

[1] A. Nicolson and K. K. Paliwal, "Deep Learning For Minimum Mean-Square Error Approaches to Speech Enhancement", Submitted with revisions to Speech Communication.
