## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

def network_selector(network_type, inp, n_outp, **kwargs):

	if kwargs['loss_fnc'] == 'BinaryCrossentropy': sigmoid_outp = True
	elif kwargs['loss_fnc'] == 'MeanSquaredError': sigmoid_outp = False
	else: raise ValueError('Invalid loss_fnc.')

	if network_type == 'MHANet':
		from deepxi.network.attention import MHANet
		network = MHANet(
			inp=inp,
			n_outp=n_outp,
			d_model=kwargs['d_model'],
			n_blocks=kwargs['n_blocks'],
			n_heads=kwargs['n_heads'],
			d_ff=kwargs['d_ff'],
			warmup_steps=kwargs['warmup_steps'],
			causal=kwargs['causal'],
			sigmoid_outp=sigmoid_outp
			)
	elif network_type == 'RDLNet':
		from dev.rdlnet import RDLNet
		network = RDLNet(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			length=kwargs['length'],
			m_1=kwargs['m_1'],
			padding=kwargs['padding'],
			unit_type=kwargs['unit_type'],
			sigmoid_outp=sigmoid_outp
			)
	elif network_type == 'ResNetV2':
		from deepxi.network.tcn import ResNetV2
		network = ResNetV2(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			d_f=kwargs['d_f'],
			k=kwargs['k'],
			max_d_rate=kwargs['max_d_rate'],
			padding=kwargs['padding'],
			unit_type=kwargs['unit_type'],
			sigmoid_outp=sigmoid_outp
			)
	elif network_type == 'ResNet':
		from deepxi.network.tcn import ResNet
		network = ResNet(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			d_f=kwargs['d_f'],
			k=kwargs['k'],
			max_d_rate=kwargs['max_d_rate'],
			padding=kwargs['padding'],
			sigmoid_outp=sigmoid_outp
			)
	elif network_type == 'ResLSTM':
		from deepxi.network.rnn import ResLSTM
		network = ResLSTM(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			sigmoid_outp=sigmoid_outp
			)
	else: raise ValueError('Invalid network type.')
	return network
