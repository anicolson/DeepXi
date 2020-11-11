% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

clear all; close all; clc

loss_csv = {
    'loss/reslstm-1.0c_dvb.csv',...
    'loss/resbilstm-1.0n_dvb.csv',...
    'loss/resnet_v1_dvb.csv',...
    'loss/resnet_v2_dvb.csv',...
    'loss/resnet_v3_dvb.csv',...
    'loss/resnet_v4_dvb.csv',...
    'loss/mhanet-1.1.2c_dvb.csv',...
    'loss/mhanet-1.1.3c_dvb.csv',...
    'loss/mhanet-1.1.4c_dvb.csv',...
    };

val_loss = false;

for i = 1:length(loss_csv)
    T = readtable(loss_csv{i});
    epoch = 1:height(T);
    subplot(1,2,1); plot(epoch, T.loss, 'LineWidth', 1); xlabel('Epoch'); ylabel('Training loss'); hold on;
    if val_loss
        subplot(1,2,2); plot(epoch, T.val_loss, 'LineWidth', 1); xlabel('Epoch'); ylabel('Validation loss'); hold on;
    end
end
legend(loss_csv);
hold off;
