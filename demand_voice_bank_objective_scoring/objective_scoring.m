% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc;

%% PARAMETERS
f_s = 16000; % sampling frequency (Hz).

%% PROCESSED (ENHANCED) SPEECH DIRECTORIES
y.dirs = {
%     'C:/Users/nic261/Outputs/DeepXi/mhanet-1.0c_demand_voice_bank/e125/y/mmse-lsa',...
%     'C:/Users/nic261/Outputs/DeepXi/mhanet-1.1c_demand_voice_bank/e125/y/mmse-lsa',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.0c_demand_voice_bank/e125/y/mmse-lsa',...
%     'C:/Users/nic261/Outputs/DeepXi/resnet-1.0n_demand_voice_bank/e125/y/mmse-lsa',...
    'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c_demand_voice_bank/e125/y/mmse-lsa',...
    'C:/Users/nic261/Outputs/DeepXi/resnet-1.1c_demand_voice_bank/e125/y/mmse-lsa',...
    
    };

%% REFERENCE (CLEAN) SPEECH DIRECTORY
s.paths = dir('C:/Users/nic261/Datasets/DEMAND_VB/clean_testset_wav/clean_testset_wav/*.wav');

%% OBJECTIVE SCORING
for i = 1:length(y.dirs)

    results = zeros(1, 8);

    split_str = strsplit(y.dirs{i}, '/');
    ver = [split_str{end-3}, '_', split_str{end-2}, '_', split_str{end}];

    for j = 1:length(s.paths)

        [s.wav, clean_speech_fs] = audioread([s.paths(j).folder, '/', s.paths(j).name]);
        [y.wav, enhanced_speech_fs] = audioread([y.dirs{i}, '/', s.paths(j).name]);
        y.wav = y.wav(1:length(s.wav));

        if any(isnan(y.wav(:))) || any(isinf(y.wav(:)))
            error('NaN or Inf value in enhanced speech.')
        end

        [CSIG, CBAK, COVL] = composite(s.wav, y.wav, f_s);
        PESQ = pesq(s.wav, y.wav, f_s);
        STOI = stoi(s.wav, y.wav, f_s);
        [SNR, SegSNR]= comp_snr(s.wav, y.wav, f_s);

        results(1) = results(1) + 1;
        results(2) = results(2) + CSIG;
        results(3) = results(3) + CBAK;
        results(4) = results(4) + COVL;
        results(5) = results(5) + PESQ;
        results(6) = results(6) + STOI;
        results(7) = results(7) + SegSNR;
        results(8) = results(8) + SNR;

        clc;
        fprintf('%.2f%%\n', 100*j/length(s.paths));
    end

    results(2) = results(2)./results(1);
    results(3) = results(3)./results(1);
    results(4) = results(4)./results(1);
    results(5) = results(5)./results(1);
    results(6) = results(6)./results(1);
    results(7) = results(7)./results(1);
    results(8) = results(8)./results(1);

    if ~exist('./results.txt', 'file')
        fileID = fopen('./results.txt', 'w');
        fprintf(fileID, 'ver, CSIG, CBAK, COVL, PESQ, STOI, SegSNR, SNR\n');
        fclose(fileID);
    end

    fileID = fopen('./results.txt', 'a');
    fprintf(fileID, '%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n', ver, ...
        results(2), results(3), results(4), ...
        results(5), results(6)*100, results(7), results(8));
    fclose(fileID);
end
% EOF
