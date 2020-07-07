% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

%
%% This needs to be updated as stats are now stored in a pickled python object.
%

clear all; close all; clc
set(0,'defaultTextInterpreter','latex');

load('data/stats.mat')
mu = stats.mu_hat;
sigma = stats.sigma_hat;
f_bins = 0:(length(mu)-1);

xi_dB = -100:100;
dist = [];
for i = 1:length(mu)
    pd = makedist('Normal','mu',mu(i),'sigma',sigma(i));
    dist = [dist; pdf(pd, xi_dB)];
end
h = surf(xi_dB, f_bins, dist); colorbar;
h.EdgeColor = 'none';
ylabel('$k$')
xlabel('$\xi_{\rm dB}[l,k]$')
grid on;
% set(gca, 'xdir', 'reverse')
set(gca, 'ydir', 'reverse')
ylim([min(f_bins) max(f_bins)])
