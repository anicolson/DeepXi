## FILE:           dct.py 
## DATE:           2019
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Short-time Discrete Cosine Transform (STDCT). DCT Type-II is used.
##
## Copyright (C) 2019  Aaron Nicolson
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools
from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.contrib.signal.python.ops import window_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import spectral_ops

def stdct(signals, frame_length, frame_step, fft_length=None,
         window_fn=functools.partial(window_ops.hann_window, periodic=True),
         pad_end=False, name=None):
  with ops.name_scope(name, 'stdct', [signals, frame_length,
                                     frame_step]):
    signals = ops.convert_to_tensor(signals, name='signals')
    signals.shape.with_rank_at_least(1)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)

    if fft_length is None:
      fft_length = _enclosing_power_of_two(frame_length)
    else:
      fft_length = ops.convert_to_tensor(fft_length, name='fft_length')

    framed_signals = shape_ops.frame(
        signals, frame_length, frame_step, pad_end=pad_end)

    if window_fn is not None:
      window = window_fn(frame_length, dtype=framed_signals.dtype)
      framed_signals *= window

    return spectral_ops.dct(framed_signals)
