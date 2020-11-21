% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc;

%% GET MATLAB_FEAT REPOSITORY
addpath('./deepxi')

%% PARAMETERS
f_s = 16000; % sampling frequency (Hz).
snr = -5:5:15; % SNR levels to test.

%% PROCESSED (ENHANCED) SPEECH DIRECTORIES
y.dirs = {
%     'C:/Users/nic261/Outputs/DeepXi/mhanet-1.1c/e200/y/mmse-lsa',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.1n/e180/y/mmse-lsa',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c/e200/y/mmse-lsa',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c/e200/y/mmse-stsa',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c/e200/y/wf',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c/e200/y/cwf',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c/e200/y/srwf',...
    'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c/e200/y/irm',...
    'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c/e200/y/ibm',...
    };

%% REFERENCE (CLEAN) SPEECH DIRECTORY
s.paths = dir('C:/Users/nic261/Datasets/deep_xi_dataset/test_clean_speech/*.wav');

%% OBJECTIVE SCORES DIRECTORY
res_dir = 'log/results/objective_scores';
if ~exist(res_dir, 'dir')
    mkdir(res_dir)
end

%% OBJECTIVE SCORING
for i = 1:length(y.dirs)

    noise_src_set = {};
    results = MapNested();

    split_str = strsplit(y.dirs{i}, '/');
    ver = [split_str{end-3}, '_', split_str{end-2}, '_', split_str{end}];

    for j = 1:length(s.paths)
        for k = snr

            s.wav = audioread([s.paths(j).folder, '/', s.paths(j).name]);

            split_basename = strsplit(s.paths(j).name, '_');
            noise_src = split_basename{end};
            snr_str = num2str(k);

            y.wav = audioread([y.dirs{i}, '/', s.paths(j).name(1:end-4), ...
                '_', snr_str, 'dB.wav']);

            y.wav = y.wav(1:length(s.wav));

            if any(isnan(y.wav(:))) || any(isinf(y.wav(:)))
                error('NaN or Inf value in enhanced speech.')
            end

            if ~any(strcmp(noise_src_set, noise_src))
                noise_src_set{end+1} = noise_src;
            end

            [CSIG, CBAK, COVL] = composite(s.wav, y.wav, f_s);
            PESQ = pesq(s.wav, y.wav, f_s);
            STOI = stoi(s.wav, y.wav, f_s);

            results = add_score(CSIG, results, noise_src, snr_str, 'CSIG');
            results = add_score(CBAK, results, noise_src, snr_str, 'CBAK');
            results = add_score(COVL, results, noise_src, snr_str, 'COVL');
            results = add_score(PESQ, results, noise_src, snr_str, 'PESQ');
            results = add_score(STOI, results, noise_src, snr_str, 'STOI');

        end
        clc;
        fprintf('%.2f%%\n', 100*j/length(s.paths));
    end

    fileID = fopen([res_dir, '/', ver, '.csv'],'w');
    fprintf(fileID, 'noise_src, snr_db, CSIG, CBAK, COVL, PESQ, STOI\n');
    avg.CSIG = [];
    avg.CBAK = [];
    avg.COVL = [];
    avg.PESQ = [];
    avg.STOI = [];
    for j = 1:length(noise_src_set)
        for k = snr

            snr_str = num2str(k);

            CSIG = mean(results(noise_src_set{j}, snr_str, 'CSIG'));
            CBAK = mean(results(noise_src_set{j}, snr_str, 'CBAK'));
            COVL = mean(results(noise_src_set{j}, snr_str, 'COVL'));
            PESQ = mean(results(noise_src_set{j}, snr_str, 'PESQ'));
            STOI = mean(results(noise_src_set{j}, snr_str, 'STOI'));

            fprintf(fileID, '%s, %s, %.2f, %.2f, %.2f, %.2f, %.2f\n', ...
                noise_src_set{j}, snr_str, ...
                CSIG, CBAK, COVL, PESQ, 100*STOI);

            avg.CSIG = [avg.CSIG; results(noise_src_set{j}, snr_str, 'CSIG')];
            avg.CBAK = [avg.CBAK; results(noise_src_set{j}, snr_str, 'CBAK')];
            avg.COVL = [avg.COVL; results(noise_src_set{j}, snr_str, 'COVL')];
            avg.PESQ = [avg.PESQ; results(noise_src_set{j}, snr_str, 'PESQ')];
            avg.STOI = [avg.STOI; results(noise_src_set{j}, snr_str, 'STOI')];

        end
    end
    fclose(fileID);

    avg_path = [res_dir, '/average.csv'];

    if ~exist(avg_path, 'file')
        fileID = fopen(avg_path, 'w');
        fprintf(fileID, 'ver, CSIG, CBAK, COVL, PESQ, STOI\n');
        fclose(fileID);
    end

    fileID = fopen(avg_path, 'a');
    fprintf(fileID, '%s, %.2f, %.2f, %.2f, %.2f, %.2f\n', ver, ...
        mean(avg.CSIG), mean(avg.CBAK), mean(avg.COVL), ...
        mean(avg.PESQ), 100*mean(avg.STOI));
    fclose(fileID);
end
% EOF
