%% FILE:           SE_TEST.m 
%% DATE:           2018
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Creates speech enhancement test set.
clear all; close all; clc;

%% SAVE PATH
save_path = '~/set/SE_TEST'; % save path.

%% OPTIONS
clean_speech_per_noise_source = 10; % number of clean speech recordings per noise source.
Q = -20:5:30; % SNR dB levels.
fs = 16000; % sampling frequency.
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', 43)); % seed.

%% FILE LISTS
s.paths = dir('/media/aaron/Filesystem/speech/librispeech/LibriSpeech/test-clean/*/*/*.flac'); % Librispeech test-clean set.
d.paths = {'/media/aaron/Filesystem/noise/RSG-10/16kHz/SIGNAL019.wav', ... % voice babble from RSG-10 (for testing).
    '/media/aaron/Filesystem/noise/UrbanSound/data/street_music/26270.wav', ... % street music 26270 from UrbanSound (for testing).
    '/media/aaron/Filesystem/noise/RSG-10/16kHz/SIGNAL020.wav',... % F16 from RSG-10 (for testing).
    '/media/aaron/Filesystem/noise/RSG-10/16kHz/SIGNAL021.wav',... % factory (welding) from RSG-10 (for testing).
};

%% SAVE DIRECTORIES
set = {'/clean_speech', '/noise', '/noisy_speech'}; % sets.
for i = 1:numel(set)
    if ~exist([save_path, set{i}], 'dir')
        mkdir([save_path, set{i}]); % create.
    else
        rmdir([save_path, set{i}], 's'); % remove directory if it exists.
        mkdir([save_path, set{i}]); % create.
    end % make training set directory.
end

p = randperm(length(s.paths)); % random index.
k = 1; c = 1; total_iterations = numel(d.paths)*clean_speech_per_noise_source*length(Q); % counts.
for i = 1:numel(d.paths)
    [d.src, fs_ori] = audioread(d.paths{i});  
    if size(d.src, 2) ~= 1 % if more than one channel.
        d.src = mean(d.src, 2);
    end
    if fs_ori > fs % if original frequency is higher than 16000 Hz.
        [A, B] = rat(fs_ori/fs);
        d.src = resample(d.src, B, A);
    elseif fs_ori <  fs
        error('Incorrect sampling frequency.')
    end
    for j = 1:clean_speech_per_noise_source
        [s.wav, fs_ori] = audioread([s.paths(p(k)).folder, '/', s.paths(p(k)).name]);
        if fs_ori ~=  fs
            error('Incorrect sampling frequency.')
        end
        s.N = length(s.wav); 
        d.N = length(d.src); 
        d.R = randi(1 + d.N - s.N);  % generate random start location in noise waveform.
        d.wav = d.src(d.R:d.R + s.N - 1); % noise waveform.
        utterance = s.paths(p(k)).name(1:end-5); % get speaker.
        [~, noise_source, ~] = fileparts(d.paths{i}); % get speaker.
        audiowrite([save_path, '/clean_speech/', utterance, '_', noise_source, '.wav'], s.wav(:), fs);
        audiowrite([save_path, '/noise/', utterance, '_', noise_source, '.wav'], d.wav(:), fs);        
        for q = Q
            if length(s.wav) ~= length(d.wav)
                error('Length of noise segment is not equal to length of clean speech recording.')
            end
            [x.wav, ~] = addnoise(s.wav(:), d.wav(:), q); % noisy speech.
            audiowrite([save_path, '/noisy_speech/', utterance, '_', noise_source, '_', num2str(q), 'dB.wav'], x.wav(:), fs);
            clc;
            fprintf('Creating test set: %3.2f%% complete.\n', 100*(c/total_iterations));
            c = c + 1;
        end
        k = k + 1;
    end
end