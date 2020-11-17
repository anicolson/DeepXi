## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

def network_selector(network_type, inp, n_outp, **kwargs):
	if network_type == 'MHANetV3':
		from deepxi.network.attention import MHANetV3
		network = MHANetV3(
			inp=inp,
			n_outp=n_outp,
			d_model=kwargs['d_model'],
			n_blocks=kwargs['n_blocks'],
			n_heads=kwargs['n_heads'],
			warmup_steps=kwargs['warmup_steps'],
			max_len=kwargs['max_len'],
			causal=kwargs['causal'],
			outp_act=kwargs['outp_act']
			)
	elif network_type == 'MHANetV2':
		from deepxi.network.attention import MHANetV2
		network = MHANetV2(
			inp=inp,
			n_outp=n_outp,
			d_model=kwargs['d_model'],
			n_blocks=kwargs['n_blocks'],
			n_heads=kwargs['n_heads'],
			warmup_steps=kwargs['warmup_steps'],
			causal=kwargs['causal'],
			outp_act=kwargs['outp_act']
			)
	elif network_type == 'MHANet':
		from dev.new_attention import MHANet
		network = MHANet(
			inp=inp,
			n_outp=n_outp,
			d_model=kwargs['d_model'],
			n_blocks=kwargs['n_blocks'],
			n_heads=kwargs['n_heads'],
			warmup_steps=kwargs['warmup_steps'],
			causal=kwargs['causal'],
			outp_act=kwargs['outp_act']
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
			outp_act=kwargs['outp_act']
			)
	elif network_type == 'ResNetV4':
		from deepxi.network.tcn import ResNetV4
		network = ResNetV4(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			d_f=kwargs['d_f'],
			k=kwargs['k'],
			max_d_rate=kwargs['max_d_rate'],
			padding=kwargs['padding'],
			unit_type=kwargs['unit_type'],
			outp_act=kwargs['outp_act']
			)
	elif network_type == 'ResNetV3':
		from deepxi.network.tcn import ResNetV3
		network = ResNetV3(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			d_f=kwargs['d_f'],
			k=kwargs['k'],
			max_d_rate=kwargs['max_d_rate'],
			padding=kwargs['padding'],
			unit_type=kwargs['unit_type'],
			outp_act=kwargs['outp_act']
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
			outp_act=kwargs['outp_act']
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
			outp_act=kwargs['outp_act']
			)
	elif network_type == 'ResBiLSTM':
		from deepxi.network.rnn import ResBiLSTM
		network = ResBiLSTM(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			outp_act=kwargs['outp_act']
			)
	elif network_type == 'ResLSTM':
		from deepxi.network.rnn import ResLSTM
		network = ResLSTM(
			inp=inp,
			n_outp=n_outp,
			n_blocks=kwargs['n_blocks'],
			d_model=kwargs['d_model'],
			outp_act=kwargs['outp_act']
			)
	else: raise ValueError('Invalid network type.')
	return network
