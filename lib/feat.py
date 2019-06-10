## FILE:           feat.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Functions to create feature extraction graph.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

# import dct
import numpy as np
from scipy import signal
import tensorflow as tf

def stft(x, Nw, Ns, NFFT):
    '''
    Computes the single-sided short-time Fourier using the Hamming window.
    Includes the DC component (0), and the Nyquist frequency component (NFFT/2 + 1).
        
    Input: 
        x - waveform.
        Nw - window length (samples).
        Ns - window shift (samples).
        NFFT - number of DFT bins.
        
    Output: 
        STFT - single-sided STFT.
    '''
    return tf.contrib.signal.stft(x, Nw, Ns, NFFT, 
        tf.contrib.signal.hamming_window, True) # single-sided STFT.

def stps(x, Nw, Ns, NFFT):
    '''
    Computes the single-sided short-time phase spectrum using the Hamming window.
    Includes the DC component (0), and the Nyquist frequency component (NFFT/2 + 1).
        
    Input: 
        x - waveform.
        Nw - window length (samples).
        Ns - window shift (samples).
        NFFT - number of DFT bins.
        
    Output: 
        STPS - single-sided short-time phase spectrum.
    '''
    return tf.angle(tf.contrib.signal.stft(x, Nw, Ns, NFFT, 
        tf.contrib.signal.hamming_window, True)) # single-sided phase spectrum.

def stms(x, Nw, Ns, NFFT):
    '''
    Computes the single-sided short-time magnitude spectrum using the Hamming window.
    Includes the DC component (0), and the Nyquist frequency component (NFFT/2 + 1).
        
    Input: 
        x - waveform.
        Nw - window length (samples).
        Ns - window shift (samples).
        NFFT - number of DFT bins.
        
    Output: 
        STMS - single-sided short-time magnitude spectrum.
    '''
    return tf.abs(tf.contrib.signal.stft(x, Nw, Ns, NFFT, 
        tf.contrib.signal.hamming_window, True)) # single-sided magnitude spectrum.

def stas(x, Nw, Ns, NFFT):
    '''
    Computes the short-time amplitude spectrum using the Hamming window.
       
    Input: 
        x - waveform.
        Nw - window length (samples).
        Ns - window shift (samples).
        NFFT - number of DFT bins (dummy variable).
        
    Output: 
        AS - amplitude spectrum.
    '''
    return tf.abs(dct.stdct(x, Nw, Ns, NFFT, 
        tf.contrib.signal.hamming_window, True)) # amplitude spectrum.

def stdct(x, Nw, Ns, NFFT):
    '''
    Computes the short-time amplitude spectrum using the Hamming window.
       
    Input: 
        x - waveform.
        Nw - window length (samples).
        Ns - window shift (samples).
        NFFT - number of DFT bins (dummy variable).
        
    Output: 
        AS - amplitude spectrum.
    '''
    return dct.stdct(x, Nw, Ns, NFFT, 
        tf.contrib.signal.hamming_window, True) # amplitude spectrum.

def psd(x, Nw, Ns, NFFT, fs):
    '''
    Computes the single-sided Power Spectral Density (PSD) using the Hamming window.
    Includes the DC component (0), and the Nyquist frequency component (NFFT/2).
    All components of the single-sided PSD are multiplied by 2, except for the DC 
    component and the Nyquist frequency component. This is to conserve the total 
    power when going from a two-sided spectrum, to a single-sided spectrum.

    Input: 
        x - waveforms.
        Nw - window length (samples).
        Ns - window shift (samples).
        NFFT - number of DFT bins.
        fs - sampling frequency.
        
    Output: 
        PSD - single-sided power spectral density.
    '''
    MS = stms(x, Nw, Ns, NFFT) # single-sided magnitude spectrum.
    PSD = tf.div(tf.square(MS), tf.to_float(Nw*fs)) # single-sided power spectral density.
    d0 = tf.shape(PSD)[0] # PSD dimension 0. 
    d1 = tf.shape(PSD)[1] # PSD dimension 1. 
    d2 = tf.shape(PSD)[2] # PSD dimension 2. 
    return tf.concat([tf.slice(PSD, [0, 0, 0], [d0, d1, 1]), tf.scalar_mul(2, 
        tf.slice(PSD, [0, 0, 1], [d0, d1, d2 - 2])), tf.slice(PSD, [0, 0, 
        tf.subtract(d2, 1)], [d0, d1, 1])], 2) # single-sided power spectral density.

def psd(x, Nw, Ns, NFFT, fs):
	'''
	Computes the single-sided Power Spectral Density (PSD) using the Hamming window.
	Includes the DC component (0), and the Nyquist frequency component (NFFT/2).
	All components of the single-sided PSD are multiplied by 2, except for the DC 
	component and the Nyquist frequency component. This is to conserve the total 
	power when going from a two-sided spectrum, to a single-sided spectrum.

	Input: 
		x - waveforms.
		Nw - window length (samples).
		Ns - window shift (samples).
		NFFT - number of DFT bins.
		fs - sampling frequency.
        
	Output: 
		PSD - single-sided power spectral density.
	'''
	MS = stms(x, Nw, Ns, NFFT) # single-sided magnitude spectrum.
	PSD = tf.div(tf.square(MS), tf.to_float(Nw*fs)) # single-sided power spectral density.
	d0 = tf.shape(PSD)[0] # PSD dimension 0. 
	d1 = tf.shape(PSD)[1] # PSD dimension 1. 
	d2 = tf.shape(PSD)[2] # PSD dimension 2. 
	return tf.concat([tf.slice(PSD, [0, 0, 0], [d0, d1, 1]), tf.scalar_mul(2, 
		tf.slice(PSD, [0, 0, 1], [d0, d1, d2 - 2])), tf.slice(PSD, [0, 0, 
		tf.subtract(d2, 1)], [d0, d1, 1])], 2) # single-sided power spectral density.

def lsse(x, H, Nw, Ns, NFFT, fs):
    '''
    Computes Log Spectral Subband Energies (LSSE) using a signal's Power Spectral
    Density (PSD) and a mel filterbank.

    Input: 
        x - waveform.
        H - mel filterbank.
        Nw - window length (samples).
        Ns - window shift (samples).
        NFFT - number of DFT bins.
        fs - sampling frequency.
        
    Output: 
        LSSE - log spectral subband energies.
    '''
    PSD = psd(x, Nw, Ns, NFFT, fs) # power spectral density.
    H = tf.tile(tf.expand_dims(H, axis=0), [tf.shape(PSD)[0],1,1]) # 3D mel-scale filterbank.
    LSSE = tf.log(tf.matmul(PSD, H, False, True)) # log spectral subband energy.
    return tf.where(tf.is_inf(LSSE), tf.zeros_like(LSSE), LSSE)

def hz2mel(f):
    '''
	Converts a value from the Hz scale to a value in the Mel scale.
		
	Input: 
		f - Hertz value.
		
	Output: 
		m - mel value.
    '''
    return 2595*np.log10(1 + (f/700))

def mel2hz(m):
    '''
	converts a value from the mel scale to a value in the Hz scale.
		
	Input: 
		m - mel value.
		
	Output: 
		f - Hertz value.
    '''
    return 700*((10**(m/2595)) - 1)

def bpoint(m, M, NFFT, fs, fl, fh):
    '''
    detirmines the frequency bin boundary point for a filterbank.
    
    Inputs:
        m - filterbank.
        M - total filterbanks.
        NFFT - number of frequency bins.
        fs - sampling frequency.
        fl - lowest frequency.
        fh - highest frequency.

    Output:
        f - frequency bin boundary point.
    '''
    return ((2*NFFT)/fs)*mel2hz(hz2mel(fl) + m*((hz2mel(fh) - hz2mel(fl))/(M + 1))) # boundary point.
    
def melfbank(M, NFFT, fs):
    '''
    creates triangular mel filter banks.

    Inputs:
        M - number of filterbanks.
        NFFT - is the length of each filter (NFFT/2 + 1 typically).
        fs - sampling frequency.

    Outputs:
        H - triangular mel filterbank matrix.

    Reference: 
        Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
        A guide to theory, algorithm, and system development. 
        Prentice Hall, Upper Saddle River, NJ, USA (pp. 315).
    '''
    fl = 0 # lowest frequency (Hz).
    fh = fs/2 # highest frequency (Hz).
    NFFT = int(NFFT) # ensure integer.
    H = np.zeros([M, NFFT], dtype=np.float32) # mel filter bank.
    for m in range(1, M + 1):
        bl = bpoint(m - 1, M, NFFT, fs, fl, fh) # lower boundary point, f(m - 1) for m-th filterbank.
        c = bpoint(m, M, NFFT, fs, fl, fh) # m-th filterbank centre point, f(m).
        bh = bpoint(m + 1, M, NFFT, fs, fl, fh) # higher boundary point f(m + 1) for m-th filterbank.
        for k in range(NFFT):
            if k >= bl and k <= c:
                H[m-1,k] = (k - bl)/(c - bl) # m-th filterbank up-slope. 
            if k >= c and k <= bh:
                H[m-1,k] = (bh - k)/(bh - c) # m-th filterbank down-slope. 
    return H

def snr(x, y):
    '''
    Finds the Signal to Noise Ratio (SNR) between the clean and noisy waveforms.

    Inputs:
        x - clean waveform.
        y - noisy waveform.

    Output:
        SNR - SNR value.
    '''
    return np.multiply(10, np.log10(np.divide(np.sum(np.square(x)), 
		np.sum(np.square(np.subtract(y,x))))))

def addnoisepad(x, d, x_len, d_len, Q, P, nconst):
	'''
	Calls addnoise() and pads the waveforms to the length given by P.
	Also normalises the waveforms using nconst.

	Inputs:
		x - clean waveform.
		d - noise waveform.
		x_len - length of x.
		d_len - length of d.
		Q - SNR level.
		P - padded length.
		nconst - normalisation constant.

	Outputs:
		x - padded clean waveform.
		y - padded noisy waveform.
		d - truncated, scaled, and padded noise waveform.
	'''
	x = tf.div(tf.cast(tf.slice(x, [0], [x_len]), tf.float32), nconst) # remove padding and normalise.
	d = tf.div(tf.cast(tf.slice(d, [0], [d_len]), tf.float32), nconst) # remove padding and normalise.
	(y, d) = addnoise(x, d, Q) # compute noisy waveform.
	total_zeros = tf.subtract(P, tf.shape(x)[0]) # number of zeros to add to each waveform.
	x = tf.pad(x, [[0, total_zeros]], "CONSTANT") # pad clean.
	y = tf.pad(y, [[0, total_zeros]], "CONSTANT") # pad noisy.
	d = tf.pad(d, [[0, total_zeros]], "CONSTANT") # pad noise.
	return (x, y, d)

def addnoise(x, d, Q):
	'''
	Adds noise to the clean waveform at a specific SNR value. A random section 
	of the noise waveform is used.

	Inputs:
		x - clean waveform.
		d - noise waveform.
		Q - SNR level.

	Outputs:
		y - noisy waveform.
		d - truncated and scaled noise waveform.
	'''
	x_len = tf.shape(x)[0] # length of clean waveform.
	d_len = tf.shape(d)[0] # length of noise waveform.
	i = tf.random_uniform([1], 0, tf.add(1, tf.subtract(d_len, x_len)), tf.int32)
	d = tf.slice(d, [i[0]], [x_len]) # extract random section of noise waveform.
	d = tf.multiply(tf.div(d, tf.norm(d)), tf.div(tf.norm(x), tf.pow(10.0, tf.multiply(0.05, Q)))) # scale the noise w.r.t. the target SNR level (Q).
	y = tf.add(x, d)  # generate the noisy waveform.
	return (y, d)

#def nframes(N, Nw, Ns):
#	'''
#	Returns the number of frames for a given sequence length, and
#	frame shift.

#	Inputs:
#		N - sequence length (samples).
#		Nw - frame width (samples).
#		Ns - frame shift (samples).

#	Output:
#		F - number of frames
#	'''
#	return tf.add(1, tf.to_int32(tf.ceil(tf.div(tf.subtract(tf.to_float(N), tf.to_float(Nw)), tf.to_float(Ns))))) # number of frames.

def nframes(N, Ns):
	'''
	Returns the number of frames for a given sequence length, and
	frame shift.

	Inputs:
		N - sequence length (samples).
		Ns - frame shift (samples).

	Output:
		F - number of frames
	'''
	return tf.to_int32(tf.ceil(tf.div(tf.to_float(N),tf.to_float(Ns)))) # number of frames.
