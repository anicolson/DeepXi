% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc

% TO DO: REVERSE Y-AXIS TICKS!

%
% In main.py, set deepxi.train(..., save_example=True) to get .mat training 
% mini-batch example.
%

load('inp_batch.mat')
load('tgt_batch.mat')
load('seq_mask_batch.mat')

%% PLOT
for i = 1:size(inp_batch, 1)
    figure (i)
       
    % Observation/input. 
    inp = rot90(squeeze(inp_batch(i,:,:)));

    % Target: mapped a priori SNR. 
    tgt = rot90(squeeze(tgt_batch(i,:,:)));

    % Sequence mask.
    seq_mask = seq_mask_batch(i,:,:);
       
    subplot(3,1,1); imagesc(flipud(inp)); colorbar;
    xlabel('Time-frame bin')
    ylabel('Frequency bin')
    title('Observation')
    set(gca,'YDir','normal')

    subplot(3,1,2); imagesc(seq_mask); colorbar;
    xlabel('Time-frame bin')
    title('Sequence mask')
    colorbar('XTick', [0, 1]);
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    
    subplot(3,1,3); imagesc(flipud(tgt)); colorbar;
    xlabel('Time-frame bin')
    ylabel('Frequency bin')
    title('Target')
    set(gca,'YDir','normal')

    set(gcf, 'Position', get(0, 'Screensize'));
end
