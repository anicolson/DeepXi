% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

function results = add_score(score, results, noise_src, snr, metric)
    if results.isKey(noise_src, snr, metric)
        results(noise_src, snr, metric) = [score; results(noise_src, snr, metric)];
    else
        results(noise_src, snr, metric) = score;
    end
end
% EOF
