% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc

load(' ')
mu = stats.mu_hat;
sigma = stats.sigma_hat;

xi_dB = -75:75;
dist = [];
for i = 1:length(mu)
    pd = makedist('Normal','mu',mu(i),'sigma',sigma(i));
    dist = [dist; pdf(pd, xi_dB)];
end
imagesc(1:length(mu), xi_dB, rot90(dist)); colorbar;
xlabel('Frequency bin')
ylabel('Instantaneous {\it a priori} SNR (dB)')
