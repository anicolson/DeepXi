# DeepXi: Residual Network-based A Priori SNR estimator

![](./fig.png "My crude attempt at drawing Renee French's Go Gopher.")


DeepXi is a Residual Network-based (ResNet) A Priori SNR estimator that was proposed in [1]. DeepXi is implemented in [TensorFlow](https://www.tensorflow.org/). 

## Prerequisites
* [TensorFlow](https://www.tensorflow.org/)
* [Python 3](https://www.python.org/)
* [MATLAB](https://www.mathworks.com/products/matlab.html)

## File Description
File | Description
--------| -----------  
train.py | Training, must give paths to the clean speech and noise training files.
inf.py | Inference, outputs .mat MATLAB a priori SNR estimates.
run.py | Used to pass variables to inf.py. must give paths to the model, and the clean speech and noise testing files.

## Availability
The DeepXi implementation will be made available after submission.

## References
[1] A. Nicolson and K. K. Paliwal, "A Priori Signal-to-Noise Ratio Estimation Using a Deep Residual Long Short-Term Memory Network", to be submitted.
