% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc;

%% SAVE PATH
save_path = '~/set/SE_TRAIN'; % save path.

%% OPTIONS
num_val_files = 1000; % number of files validation set.
Q = -10:1:20; % SNR dB levels.
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', 43)); % seed.
fs = 16000; % sampling frequency.
max_len = 30; % maximum length for noise signals (seconds).
min_len = 2; % minimum length for noise signals (seconds).

%% CLEAN SPEECH PATHS
s.paths = dir('/media/aaron/Filesystem/speech/VCTK/wav48/*/*.wav'); % VCTK files.
s.paths = [s.paths; dir('/media/aaron/Filesystem/speech/librispeech/LibriSpeech/train-clean-100/*/*/*.flac')]; % Librispeech files.

%% NOISE PATHS
d.paths = dir('/media/aaron/Filesystem/noise/VAD-dataset/*/*.wav'); % VAD noise dataset.
d.paths = [d.paths; dir('/media/aaron/Filesystem/noise/col_noise/16kHz/*.wav')]; % Coloured noise dataset.
d.paths = [d.paths; dir('/media/aaron/Filesystem/noise/Nonspeech/16kHz/*.wav')]; % Nonspeech dataset.
d.paths = [d.paths; dir('/media/aaron/Filesystem/noise/QUT-NOISE/Mono-16kHz/*.wav')]; % QUT-NOISE dataset.
d.paths = [d.paths; dir('/media/aaron/Filesystem/noise/RSG-10/16kHz/*.wav')]; % QUT-NOISE dataset.
d.paths = [d.paths; dir('/media/aaron/Filesystem/musan/noise/*/*.wav')]; % MUSAN noise dataset.
ext = {'wav', 'ogg', 'flac', 'aiff', 'aif', 'aifc', 'mp3', 'm4a', 'mp4'}; % '.au' is one format that is not checked.
for i = 1:numel(ext)
    d.paths = [d.paths; dir(['/media/aaron/Filesystem/noise/UrbanSound/data/*/*.', ext{i}])]; % UrbanSound noise datasets.
    d.paths = [d.paths; dir(['/media/aaron/Filesystem/noise/Freesound/set1/*/*.', ext{i}])]; % Freespeech noise datasets.
    d.paths = [d.paths; dir(['/media/aaron/Filesystem/noise/Freesound/set2/*/*.', ext{i}])]; % Freespeech noise datasets.
end

%% EXCLUDED NOISE RECORDINGS USED FOR TESTING
excluded = {'/media/aaron/Filesystem/noise/RSG-10/16kHz/SIGNAL019.wav', ... % voice babble from RSG-10 (for testing).
    '/media/aaron/Filesystem/noise/UrbanSound/data/street_music/26270.wav', ... % street music 26270 from UrbanSound (for testing).
    '/media/aaron/Filesystem/noise/RSG-10/16kHz/SIGNAL020.wav',... % F16 from RSG-10 (for testing).
    '/media/aaron/Filesystem/noise/RSG-10/16kHz/SIGNAL021.wav',... % factory (welding) from RSG-10 (for testing).
    };

%% CLEAN SPEECH FILE EXCLUSION
excluded_speakers = {'p232', 'p257'}; % these speakers are in the test set of the DEMAND VOICE BANK CORPUS.
fprintf('Number of clean speech files before exclusion %i.\n', length(s.paths));
count = 0;
for i = 1:length(s.paths)
    for j = 1:numel(excluded_speakers)
        if contains([s.paths(i-count).folder, '/', s.paths(i-count).name], excluded_speakers{j})
            s.paths(i-count) = [];
            count = count + 1;
        end
    end
end
fprintf('Number of clean speech files after exclusion %i.\n', length(s.paths));

%% VALIDATION PATHS
p = randperm(length(s.paths), num_val_files); % index of validation files.
s.val_paths = s.paths(p); % validation set.
s.paths(p) = []; % remove validation paths from training set.

%% CHECK FOR BAD NOISE FILES
disp('Checking for bad noise files...')
count = 0;
for i = 1:length(d.paths)
    try
        info = audioinfo([d.paths(i).folder, '/', d.paths(i).name]); 
        if info.SampleRate < fs
            count = count + 1;
            excluded{end + 1} = [d.paths(i).folder, '/', d.paths(i).name];
            fprintf('File %s has a sampling frequency of %i Hz, and has been excluded.\n', [d.paths(i).folder, '/', d.paths(i).name], info.SampleRate)
        end
    catch
        count = count + 1;
        excluded{end + 1} = [d.paths(i).folder, '/', d.paths(i).name];
        fprintf('File %s has caused an unexpected error, and has been excluded.\n', [d.paths(i).folder, '/', d.paths(i).name])
    end
end
fprintf('Bad files found: %i.\n', count)

%% NOISE FILE EXCLUSION
fprintf('Number of noise files before exclusion %i.\n', length(d.paths));
for i = 1:numel(excluded)
    idx = find(ismember(strcat(strcat({d.paths.folder}, '/'), {d.paths.name}), excluded{i}));
    if isempty(idx)
        error('File path %s could not be found.', excluded{i})        
    else
        d.paths(idx) = [];
    end
end
fprintf('Number of noise files after exclusion %i.\n', length(d.paths));

%% SAVE DIRECTORIES
set = {'/split_noise', '/train_clean_speech', '/train_noise', '/val_clean_speech', '/val_noise'}; % sets.
for i = 1:numel(set)
    if ~exist([save_path, set{i}], 'dir')
        mkdir([save_path, set{i}]); % create.
    else
        rmdir([save_path, set{i}], 's'); % remove directory if it exists.
        mkdir([save_path, set{i}]); % create.
    end % make training set directory.
end

%% NOISE SPLIT
for i = 1:length(d.paths)
    [d.src, fs_ori] = audioread([d.paths(i).folder, '/', d.paths(i).name]);
    switch d.paths(i).name(end-3:end)
        case '.wav'
        case {'.mp3', '.ogg', '.aif', '.m4a', '.mp4'}
            d.paths(i).name = [d.paths(i).name(1:end-4), '.wav']; % .wav file.
        case {'flac', 'aiff', 'aifc'}
            d.paths(i).name = [d.paths(i).name(1:end-5), '.wav']; % .wav file.
        otherwise
            error('Error: unknown file format for %s.', [d.paths(i).folder, '/', d.paths(i).name]);
    end
    if size(d.src, 2) ~= 1 % if more than one channel.
        d.src = mean(d.src, 2);
    end
    if fs_ori < fs % if original frequency is less than 16000 Hz.
        error('Error: sampling frequency too low for %s.', [d.paths(i).folder, '/', d.paths(i).name]);
    end
    if fs_ori > fs % if original frequency is higher than 16000 Hz.
        [A, B] = rat(fs_ori/fs);
        d.src = resample(d.src, B, A);
    end
    if length(d.src)/fs > max_len
        j = 1;
        start_idx = 1;
        end_idx = max_len*fs;
        flag = 1;
        while flag
            split_name = [save_path, '/split_noise/', d.paths(i).name(1:end-4), '_', num2str(j), '.wav'];
            d.split = d.src(start_idx:end_idx)./max(abs(d.src(start_idx:end_idx)));
            if any(isnan(d.split)) || any(isinf(d.split))
                error('Error: NaN or Inf value.\nFile path:%s.', [d.paths(i).folder, '/', d.paths(i).name])
            end
            audiowrite(split_name, d.split, fs);
            j = j + 1;           
            start_idx = start_idx + max_len*fs;
            end_idx = end_idx + max_len*fs;
            if end_idx > length(d.src)
                end_idx = length(d.src);
            end
            if start_idx > (length(d.src) - (min_len*fs))
                flag = 0;
            end
        end
    else
        d.wav = d.src./max(abs(d.src));
        if any(isnan(d.wav)) || any(isinf(d.wav))
            error('Error: NaN or Inf value.\nFile path:%s.', [d.paths(i).folder, '/', d.paths(i).name])
        end
        audiowrite([save_path, '/split_noise/', d.paths(i).name], d.wav, fs);
    end
    clc;
    fprintf('Creating noise set: %3.2f%% complete.\n', 100*(i/length(d.paths)));
end

%% SPLIT NOISE PATHS
d.paths = dir([save_path, '/', set{1}, '/*.wav']);

%% VALIDATION SET
p = randperm(length(s.val_paths)); % shuffle validation files.
j = 0; % count.
for i = p
    d.SNR = Q(randi([1, length(Q)])); % random SNR level.
	[s.wav, fs_ori] = audioread([s.val_paths(i).folder, '/', s.val_paths(i).name]); % read validation clean waveform.
    if size(s.wav, 2) ~= 1 % if more than one channel.
        s.wav = mean(s.wav, 2);
    end
    if fs_ori < fs % if original frequency is less than 16000 Hz.
        error('Error: sampling frequency too low.');
    end
    if fs_ori > fs % if original frequency is higher than 16000 Hz.
        [A, B] = rat(fs_ori/fs);
        s.wav = resample(s.wav, B, A);
    end
    s.N = length(s.wav); % length of validation clean waveform.
    d.N = 0; % initialise length search.
    while d.N < s.N
        d.i = randi([1, length(d.paths)]); % random noise file.
        [d.src, ~] = audioread([d.paths(d.i).folder, '/', d.paths(d.i).name]); % read validation noise waveform.   
        d.N = length(d.src); % length of validation noise waveform.
    end
    d.R = randi(1 + d.N - s.N);  % generate random start location in noise waveform.
    d.wav = d.src(d.R:d.R + s.N - 1); % noise waveform.    
    [~, spkr, ~] = fileparts(s.val_paths(i).folder); % get speaker.
    if any(isnan(s.wav)) || any(isinf(s.wav))
        error('Error: NaN or Inf value.\nFile path:%s.', [s.val_paths(i).folder, '/', s.val_paths(i).name])
    end
    if any(isnan(d.wav)) || any(isinf(d.wav))
        error('Error: NaN or Inf value.\nFile path:%s.', [d.paths(d.i).folder, '/', d.paths(d.i).name])
    end
    if strcmp(s.val_paths(i).name(end-3:end), '.wav') % .wav compatability.
        audiowrite([save_path, '/val_clean_speech/', spkr, '_', s.val_paths(i).name(1:end-4), ...
            '_', d.paths(d.i).name(1:end-4), '_', num2str(d.SNR), 'dB.wav'], s.wav, fs); % write validation clean waveform.
        audiowrite([save_path, '/val_noise/', spkr, '_', s.val_paths(i).name(1:end-4), ...
            '_', d.paths(d.i).name(1:end-4), '_', num2str(d.SNR), 'dB.wav'], d.wav, fs); % write validation noise waveform.
    else % .flac compatability.
        audiowrite([save_path, '/val_clean_speech/', spkr, '_', s.val_paths(i).name(1:end-5), ...
            '_', d.paths(d.i).name(1:end-4), '_', num2str(d.SNR), 'dB.wav'], s.wav, fs); % write validation clean waveform.
        audiowrite([save_path, '/val_noise/', spkr, '_', s.val_paths(i).name(1:end-5), ...
            '_', d.paths(d.i).name(1:end-4), '_', num2str(d.SNR), 'dB.wav'], d.wav, fs); % write validation noise waveform.
    end
    d.paths(d.i) = []; % remove noise signal from split paths.
    
    j = j + 1; % increment count.
    clc;
    fprintf('Creating validation set: %3.2f%% complete.\n', 100*(j/length(s.val_paths)));
end

%% NOISE TRAINING SET
for i = 1:length(d.paths)
	[d.wav, ~] = audioread([d.paths(i).folder, '/', d.paths(i).name]); % read noise training waveform.
    audiowrite([save_path, '/train_noise/', d.paths(i).name], d.wav, fs); % write training waveform.
    clc;
    fprintf('Creating noise training set: %3.2f%% complete.\n', 100*(i/length(s.paths)));
end

%% CLEAN SPEECH TRAINING SET
for i = 1:length(s.paths)
	[s.wav, fs_ori] = audioread([s.paths(i).folder, '/', s.paths(i).name]); % read training waveform.
    if size(s.wav, 2) ~= 1 % if more than one channel.
        s.wav = mean(s.wav, 2);
    end
    if fs_ori < fs % if original frequency is less than 16000 Hz.
        error('Error: sampling frequency too low.');
    end
    if fs_ori > fs % if original frequency is higher than 16000 Hz.
        [A, B] = rat(fs_ori/fs);
        s.wav = resample(s.wav, B, A);
    end
    [~, spkr, ~] = fileparts(s.paths(i).folder); % get speaker.
    if any(isnan(s.wav)) || any(isinf(s.wav))
        error('Error: NaN or Inf value.\nFile path:%s.', [s.paths(i).folder, '/', s.paths(i).name])
    end
    if strcmp(s.paths(i).name(end-3:end), '.wav') % .wav compatability.
        audiowrite([save_path, '/train_clean_speech/', spkr, '_', s.paths(i).name(1:end-4), '.wav' ], ...
        s.wav, fs); % write training waveform.
    else % .flac compatability.
        audiowrite([save_path, '/train_clean_speech/', spkr, '_', s.paths(i).name(1:end-5), '.wav' ], ...
        s.wav, fs); % write training waveform.
    end
    clc;
    fprintf('Creating clean speech training set: %3.2f%% complete.\n', 100*(i/length(s.paths)));
end
