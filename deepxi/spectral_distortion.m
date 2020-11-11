% AUTHOR:         Aaron Nicolson
% AFFILIATION:    Signal Processing Laboratory, Griffith University
%
% This Source Code Form is subject to the terms of the Mozilla Public
% License, v. 2.0. If a copy of the MPL was not distributed with this
% file, You can obtain one at http://mozilla.org/MPL/2.0/.

function D = spectral_distortion(ref, est)
% SD - Spectral Distortion (SD) in dB per frame.
%
% Input/s:
%	ref - instantaneous a priori/posteriori SNR.
%	est - estimate a priori/posteriori SNR.
%
% Output/s:
%	D - Spectral Distortion (SD) in dB per frame.

    ref = max(ref, 1e-12);
    est = max(est, 1e-12);
    ref = 10*log10(ref);
    est = 10*log10(est);
    D = sqrt((1/size(ref, 2)).*sum((ref - est).^2, 2));
end
%% EOF
