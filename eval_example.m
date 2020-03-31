% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc

%
% In main.py, set deepxi.train(..., save_example=True) to get .mat training 
% mini-batch example.
%

load('x_STMS_batch.mat')
load('xi_bar_batch.mat')
load('seq_mask_batch.mat')

for i = 1:size(x_STMS_batch, 1)
    figure (i)
    
    % Observation/input: noisy-speech short-time magnitude spectrum. 
    x_STMS = rot90(squeeze(x_STMS_batch(i,:,:)));
    x_STMS_dB = 10*log10(x_STMS);

    % Target: mapped a priori SNR. 
    xi_bar = rot90(squeeze(xi_bar_batch(i,:,:)));

    % Sequence mask.
    seq_mask = seq_mask_batch(i,:,:);
    
    subplot(2,2,1); imagesc(x_STMS); colorbar;
    xlabel('Time-frame bin')
    ylabel('Frequency bin')
    title('Noisy-speech short-time magnitude spectrum')

    subplot(2,2,2); imagesc(x_STMS_dB); colorbar;
    xlabel('Time-frame bin')
    ylabel('Frequency bin')
    title('Noisy-speech short-time magnitude spectrum in dB')

    subplot(2,2,3); imagesc(seq_mask); colorbar;
    xlabel('Time-frame bin')
    ylabel('Frequency bin')
    title('Sequence mask')
    
    subplot(2,2,4); imagesc(xi_bar); colorbar;
    xlabel('Time-frame bin')
    ylabel('Frequency bin')
    title('Mapped {\it a priori} SNR')
    
    set(gcf, 'Position', get(0, 'Screensize'));
end
