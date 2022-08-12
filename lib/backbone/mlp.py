"""The multi-layer perceptron"""
from copy import deepcopy
from typing import Union

from torch import nn


class MLP(nn.Sequential):
	def __init__(self, input_dim, output_dim, hidden_dims_lst):
		'''
			input_dim (int)
			output_dim (int)
			hidden_dims_lst (list, each element is a integer, indicating the hidden size)
		'''
		super(MLP, self).__init__()
		layer_size = len(hidden_dims_lst) + 1
		dims = [input_dim] + hidden_dims_lst + [output_dim]

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v):
		# predict
		v = v.float().to(device)
		for i, l in enumerate(self.predictor):
			v = F.relu(l(v))
		return v  

# class MLPConfig(SimpleConfig):
#     def __init__(self, mapping=None):
#         if mapping is None:
#             mapping = {}
#         instance = MLP
#         super(MLPConfig, self).__init__(mapping=mapping, instance=instance)

#     def get_parser(self, parser: Union[ArgumentParser, _ArgumentGroup, None] = None):
#         parser = super(MLPConfig, self).get_parser(parser)
#         parser = self.add_argument(parser, '--input_size', type=int, default=256,
#                                    help='The input dimension of MLP.')
#         parser = self.add_argument(parser, "--hidden_size", type=int, default=128)
#         parser = self.add_argument(parser, "--output_size", type=int, default=1)
#         parser = self.add_argument(parser, "--num_layers", type=int, default=4)
#         parser = self.add_argument(parser, "--dropout", type=float, default=0)
#         parser = self.add_argument(parser, "--activation", type=str, default="ReLU")
#         parser = self.add_argument(parser, "--bias", action="store_true", default=False)
#         parser = self.add_argument(parser, "--normalization", type=str, default=None)
#         parser = self.add_argument(parser, "--norm_before_activation", action="store_true", default=False)
#         return parser


# class MLP(nn.Module):
#     def __init__(self,
#                  input_size: int,
#                  hidden_size: int,
#                  output_size: int,
#                  num_layers: int,
#                  dropout: float,
#                  bias=True,
#                  activation=None,
#                  normalization=None,
#                  norm_before_activation=False, ):
#         super(MLP, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#         self.dropout = nn.Dropout(dropout)
#         self.bias = bias
#         self.activation = get_activation_function(activation)
#         self.normalization = normalization
#         self.norm_before_activation = norm_before_activation
#         self.nets = nn.ModuleList()
#         self.__bulid_nets()

#     def __buildhiddenblock(self, input_size, output_size, dropout, activation, normalization):
#         if self.norm_before_activation:
#             res = [
#                 normalization,
#                 activation,
#                 dropout,
#                 nn.Linear(input_size, output_size, bias=self.bias)
#             ]
#         else:
#             res = [
#                 activation,
#                 normalization,
#                 dropout,
#                 nn.Linear(input_size, output_size, bias=self.bias)
#             ]
#         return res

#     def __buildinputblock(self, input_size, output_size, dropout):
#         res = [
#             dropout,
#             nn.Linear(input_size, output_size, bias=self.bias)
#         ]
#         return res

#     def __buildoutputblock(self, input_size, output_size, dropout, activation, normalization):
#         return self.__buildhiddenblock(input_size, output_size, dropout, activation, normalization)

#     def __bulid_nets(self):
#         nets = []
#         if self.num_layers == 1:
#             nets.extend(self.__buildinputblock(self.input_size, self.output_size, self.dropout))
#         else:
#             nets.extend(self.__buildinputblock(self.input_size, self.hidden_size, self.dropout))
#             for _ in range(self.num_layers - 2):
#                 nets.extend(self.__buildhiddenblock(self.hidden_size,
#                                                     self.hidden_size,
#                                                     self.dropout,
#                                                     deepcopy(self.activation),
#                                                     deepcopy(self.normalization)))
#             nets.extend(self.__buildoutputblock(self.hidden_size,
#                                                 self.output_size,
#                                                 self.dropout,
#                                                 deepcopy(self.activation),
#                                                 deepcopy(self.normalization)))

#         nets = [x for x in nets if x is not None]
#         self.nets.extend(nets)

#     def forward(self, xin):
#         out = xin
#         for net in self.nets:
#             out = net(out)
#         return out

#     @classmethod
#     def get_instance(cls, config: MLPConfig, args, params=None):
#         assert type(config) == MLPConfig
#         kwargs = config.get_config(args, params=params)
#         return MLP(**kwargs)


# class PositionwiseFeedForward(nn.Module):
#     """Implements FFN equation."""

#     def __init__(self, d_model, d_ff, activation="ReLU", dropout=0.1, d_out=None):
#         """Initialization.

#         :param d_model: the input dimension.
#         :param d_ff: the hidden dimension.
#         :param activation: the activation function.
#         :param dropout: the dropout rate.
#         :param d_out: the output dimension, the default value is equal to d_model.
#         """
#         super(PositionwiseFeedForward, self).__init__()
#         if d_out is None:
#             d_out = d_model
#         # By default, bias is on.
#         self.W_1 = nn.Linear(d_model, d_ff)
#         self.W_2 = nn.Linear(d_ff, d_out)
#         self.dropout = nn.Dropout(dropout)
#         self.act_func = get_activation_function(activation)

#     def forward(self, x):
#         """
#         The forward function
#         :param x: input tensor.
#         :return:
#         """
#         return self.W_2(self.dropout(self.act_func(self.W_1(x))))