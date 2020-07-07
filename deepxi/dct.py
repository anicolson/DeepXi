## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.ops.signal import reconstruction_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.util.tf_export import tf_export

def stdct(signals, frame_length, frame_step, fft_length=None,
         window_fn=window_ops.hann_window,
         pad_end=False, name=None):
  """
  Short-time discrete cosine transform.

  Argument/s:

  Returns:
  """
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

    # Optionally window the framed signals.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=framed_signals.dtype)
      framed_signals *= window

    return dct_ops.dct(framed_signals, n=fft_length)

def inverse_stdct(stdcts,
                 frame_length,
                 frame_step,
                 fft_length=None,
                 window_fn=window_ops.hann_window,
                 name=None):
  """
	Inverse short-time discrete cosine transform.

	Argument/s:

	Returns:
  """
  with ops.name_scope(name, 'inverse_stdct', [stdcts]):
    stdcts = ops.convert_to_tensor(stdcts, name='stdcts')
    stdcts.shape.with_rank_at_least(2)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)
    if fft_length is None:
      fft_length = _enclosing_power_of_two(frame_length)
    else:
      fft_length = ops.convert_to_tensor(fft_length, name='fft_length')
      fft_length.shape.assert_has_rank(0)

    frames = dct_ops.idct(stdcts, n=fft_length)

    # frame_length may be larger or smaller than fft_length, so we pad or
    # truncate frames to frame_length.
    frame_length_static = tensor_util.constant_value(frame_length)
    # If we don't know the shape of frames's inner dimension, pad and
    # truncate to frame_length.
    if (frame_length_static is None or frames.shape.ndims is None or
        frames.shape.as_list()[-1] is None):
      frames = frames[..., :frame_length]
      frames_rank = array_ops.rank(frames)
      frames_shape = array_ops.shape(frames)
      paddings = array_ops.concat(
          [array_ops.zeros([frames_rank - 1, 2],
                           dtype=frame_length.dtype),
           [[0, math_ops.maximum(0, frame_length - frames_shape[-1])]]], 0)
      frames = array_ops.pad(frames, paddings)
    # We know frames's last dimension and frame_length statically. If they
    # are different, then pad or truncate frames to frame_length.
    elif frames.shape.as_list()[-1] > frame_length_static:
      frames = frames[..., :frame_length_static]
    elif frames.shape.as_list()[-1] < frame_length_static:
      pad_amount = frame_length_static - frames.shape.as_list()[-1]
      frames = array_ops.pad(frames,
                                  [[0, 0]] * (frames.shape.ndims - 1) +
                                  [[0, pad_amount]])

    # The above code pads the inner dimension of frames to frame_length,
    # but it does so in a way that may not be shape-inference friendly.
    # Restore shape information if we are able to.
    if frame_length_static is not None and frames.shape.ndims is not None:
      frames.set_shape([None] * (frames.shape.ndims - 1) +
                            [frame_length_static])

    # Optionally window and overlap-add the inner 2 dimensions of frames
    # into a single [samples] dimension.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=stdcts.dtype.real_dtype)
      frames *= window
    return reconstruction_ops.overlap_and_add(frames, frame_step)
