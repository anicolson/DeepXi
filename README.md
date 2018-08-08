# DeepXi: Residual Network A Priori SNR estimator
DeepXi is a Residual Network-based (ResNet) A Priori SNR estimator implemented in [TensorFlow](https://www.tensorflow.org/).

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

## References
[1] A. Nicolson and K. K. Paliwal, "A Priori Signal-to-Noise Ratio Estimation Using a Deep Residual Long Short-Term Memory Network", to be submitted.
