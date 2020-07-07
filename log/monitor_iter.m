% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc

ver = {''};

for i = 1:length(ver)
    T = readtable(['./iter/', ver{i}, '.csv']);
    iter = 1:height(T);
    plot(iter, T.loss); hold on;
    xlabel('Iteration');
    ylabel('Loss');
end
legend(ver);
hold off;