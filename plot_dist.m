%% FILE:           plot_dist.m 
%% DATE:           2018
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Plots the independant distributions for each local a priori SNR component.

xi_local_db = -65:55; % local a priori SNR in dB.
dist = [];
for i = 1:length(mu)
    pd = makedist('Normal','mu',mu(i),'sigma',sigma(i));
    dist = [dist; pdf(pd, xi_local_db)];
end
imagesc(xi_local_db, 1:length(mu), dist); colorbar;


