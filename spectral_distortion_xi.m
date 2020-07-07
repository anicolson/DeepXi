% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc;

%% GET MATLAB_FEAT REPOSITORY
addpath('~/Dropbox/GitHub/matlab_feat/feat')
addpath('./deepxi')

%% PARAMETERS
T_d = 32; % window duration (ms).
T_s = 16; % window shift (ms).
f_s = 16000; % sampling frequency (Hz).
s.N_d = round(f_s*T_d*0.001); % window duration (samples).
s.N_s = round(f_s*T_s*0.001); % window shift (samples).
s.f_s = f_s; % sampling frequency (Hz).
s.NFFT = 2^nextpow2(s.N_d); % frequency bins (samples).
d = s; x = s;
SNR_avg = -5:5:15; % SNR levels used to compute average SD level.

%% DIRECTORIES
s.dir = '/home/aaron/set/deep_xi_test_set/test_clean_speech';
d.dir = '/home/aaron/set/deep_xi_test_set/test_noise';
xi_hat_dir = input('xi_hat path:', 's');
xi_hat_dir_split = strsplit(xi_hat_dir, '/');
ver = [xi_hat_dir_split{end-2}, '_', xi_hat_dir_split{end-1}];

%% FILE LISTS
xi_hat_paths = dir([xi_hat_dir, '/*.mat']); % noise file paths.

results = MapNested();
noise_src_set = {};
SNR_set = {};
for i = 1:length(xi_hat_paths)

    load([xi_hat_paths(i).folder, '/', xi_hat_paths(i).name])

    if any(isnan(xi_hat(:))) || any(isinf(xi_hat(:)))
        error('NaN or Inf value in xi_hat.')
    end

    split_basename = strsplit(xi_hat_paths(i).name,'_');
    noise_src = split_basename{end-1};
    SNR = split_basename{end};
    clean_speech = extractBefore(xi_hat_paths(i).name, ['_', noise_src, '_', SNR]);
    SNR = SNR(1:end-6);

    s.wav = audioread([s.dir, '/', clean_speech, '_', noise_src, '.wav']); % clean speech.
    d.src = audioread([d.dir, '/', clean_speech, '_', noise_src, '.wav']); % noise.
    [~, d.wav] = add_noise(s.wav, d.src(1:length(s.wav)), str2double(SNR)); % noisy speech.

    s = analysis_stft(s, 'polar'); % clean-speech STMS.
    d = analysis_stft(d, 'polar'); % noise STMS.

    xi = (s.STMS.^2)./(d.STMS.^2); % instantaneous a priori SNR.

    xi_hat = xi_hat(1:size(xi, 1), :);
    D = spectral_distortion(xi, xi_hat);

    if any(isnan(D(:))) || any(isinf(D(:)))
        error('NaN or Inf value in D.')
    end

    if ~any(strcmp(SNR_set, SNR))
        SNR_set{end+1} = SNR;
    end

    if ~any(strcmp(noise_src_set, noise_src))
        noise_src_set{end+1} = noise_src;
    end

    if results.isKey(noise_src, SNR)
        results(noise_src, SNR) = [D; results(noise_src, SNR)];
    else
        results(noise_src, SNR) = D;
    end
    clc;
    fprintf('%.2f%%\n', 100*i/length(xi_hat_paths));
end

% there has to be a better way to do this.
for i=1:length(SNR_set)
    SNR_set{i} = str2num(SNR_set{i});
end
tmp = sort(cell2mat(SNR_set));
for i=1:length(SNR_set)
    SNR_set{i} = num2str(tmp(i));
end

res_dir = 'log/results/spectral_distortion_xi';
if ~exist(res_dir, 'dir')
    mkdir(res_dir)
end

fileID = fopen([res_dir, '/', ver, '.csv'],'w');
fprintf(fileID, 'noise, snr_db, SD\n');
avg = [];
for i=1:length(noise_src_set)
    for j=1:length(SNR_set)
        D = mean(results(noise_src_set{i}, SNR_set{j}));
        fprintf('%s, %s: %.2f.\n', noise_src_set{i}, SNR_set{j}, D);
        fprintf(fileID, '%s, %s, %.2f\n', noise_src_set{i}, SNR_set{j}, D);
        if ismember(j, SNR_avg)
            avg = [D; results(noise_src_set{i}, SNR_set{j})];
        end
    end
end
fclose(fileID);

avg_path = [res_dir, '/average.csv'];

if ~exist(avg_path, 'file')
    fileID = fopen(avg_path, 'w');
    fprintf(fileID, 'ver, SD\n');
    fclose(fileID);
end

fileID = fopen(avg_path, 'a');
fprintf(fileID, '%s, %.2f\n', ver, mean(avg));
fclose(fileID);
