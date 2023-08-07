"""
    Based on caffemodel2pytorch/caffemodel2pytorch.py from https://github.com/vadimkantorov/caffemodel2pytorch
    which is made available under the MIT License (see caffemodel2pytorch/README.md)
"""

import os
import sys
import tempfile
import subprocess
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

import google.protobuf.descriptor
import google.protobuf.descriptor_pool
import google.protobuf.symbol_database
import google.protobuf.text_format
from google.protobuf.descriptor import FieldDescriptor as FD

import ast
from torchvision.ops import RoIPool
from complex_layers import ProposalLayerModule


TRAIN = 0
TEST = 1

RPN_PRE_NMS_TOP_N = 6000
RPN_POST_NMS_TOP_N = 300
RPN_NMS_THRESH = 0.7
RPN_MIN_SIZE = 16

caffe_pb2 = None


def initialize(caffe_proto, codegen_dir=tempfile.mkdtemp(), shadow_caffe=True):
    global caffe_pb2
    if caffe_pb2 is None:
        local_caffe_proto = os.path.join(codegen_dir, os.path.basename(caffe_proto))
        with open(local_caffe_proto, 'w') as f:
            mystr = open(caffe_proto, 'rt').read()
            f.write(mystr)
        subprocess.check_call(['protoc', '--proto_path', os.path.dirname(local_caffe_proto), '--python_out',
                               codegen_dir, local_caffe_proto])
        sys.path.insert(0, codegen_dir)
        old_pool = google.protobuf.descriptor._message.default_pool
        old_symdb = google.protobuf.symbol_database._DEFAULT
        google.protobuf.descriptor._message.default_pool = google.protobuf.descriptor_pool.DescriptorPool()
        google.protobuf.symbol_database._DEFAULT = google.protobuf.symbol_database.SymbolDatabase(
            pool=google.protobuf.descriptor._message.default_pool)
        import caffe_pb2 as caffe_pb2
        google.protobuf.descriptor._message.default_pool = old_pool
        google.protobuf.symbol_database._DEFAULT = old_symdb
        sys.modules[__name__ + '.proto'] = sys.modules[__name__]
        if shadow_caffe:
            sys.modules['caffe'] = sys.modules[__name__]
            sys.modules['caffe.proto'] = sys.modules[__name__]
    return caffe_pb2


class Net(nn.Module):
    def __init__(self, prototxt, device, *args, **kwargs):
        super(Net, self).__init__()
        self.device = device
        # to account for both constructors, see https://github.com/BVLC/caffe/blob/master/python/caffe/test/test_net.py#L145-L147
        caffe_proto = kwargs.pop('caffe_proto', None)
        weights = kwargs.pop('weights', None)
        phase = kwargs.pop('phase', None)
        weights = weights or (args + (None, None))[0]
        phase = phase or (args + (None, None))[1]

        self.net_param = initialize(caffe_proto).NetParameter()
        google.protobuf.text_format.Parse(open(prototxt).read(), self.net_param)

        # Keep track of num_outputs so they can be retrieved from "bottom"
        layer_num_outputs = dict()
        for layer in list(self.net_param.layer) + list(self.net_param.layers):
            layer_type = layer.type if layer.type != 'Python' else layer.python_param.layer
            if isinstance(layer_type, int):
                layer_type = layer.LayerType.Name(layer_type)
            module_constructor = ([v for k, v in modules.items()
                                   if k.replace('_', '').upper() in [layer_type.replace('_', '').upper(),
                                                                     layer.name.replace('_', '').upper()]] + [None])[0]
            if module_constructor is not None:
                param = [v for f, v in layer.ListFields() if f.name.endswith('_param')]
                if layer_type == "ProposalLayer":
                    param = str(param[0]).split("param_str: ")[1]
                    param = "{ " + param[1:-2].replace('\\n', ',').replace('\\', '') + " }"
                    param = ast.literal_eval(param)
                    param.update({'pre_nms_topN': RPN_PRE_NMS_TOP_N,
                                  'post_nms_topN': RPN_POST_NMS_TOP_N,
                                  'nms_thresh': RPN_NMS_THRESH,
                                  'min_size': RPN_MIN_SIZE,
                                  'device': self.device})
                else:
                    param = to_dict((param + [None])[0])

                caffe_input_variable_names = list(layer.bottom)
                caffe_output_variable_names = list(layer.top)
                caffe_loss_weight = (list(layer.loss_weight) or [1.0 if layer_type.upper().endswith('LOSS')
                                                                 else 0.0]) * len(layer.top)
                caffe_propagate_down = list(getattr(layer,
                                                    'propagate_down', [])) or [True] * len(caffe_input_variable_names)
                caffe_optimization_params = to_dict(layer.param)
                if 'inplace' not in param:
                    param['inplace'] = len(caffe_input_variable_names) == 1 and caffe_input_variable_names == caffe_output_variable_names
                if 'num_input' not in param:
                    for bottom_name in caffe_input_variable_names:
                        if bottom_name in layer_num_outputs:
                            param['num_input'] = layer_num_outputs[bottom_name]
                            break
                if 'num_output' in param:
                    layer_num_outputs[layer.name] = param['num_output']
                module = module_constructor(param)
                self.add_module(layer.name,
                                module if isinstance(module, nn.Module) else CaffePythonLayerModule(module,
                                                                                                    caffe_input_variable_names,
                                                                                                    caffe_output_variable_names,
                                                                                                    param.get('param_str', '')) if type(module).__name__.endswith('Layer') else FunctionModule(module))
                module = getattr(self, layer.name)
                module.caffe_layer_name = layer.name
                module.caffe_layer_type = layer_type
                module.caffe_input_variable_names = caffe_input_variable_names
                module.caffe_output_variable_names = caffe_output_variable_names
                module.caffe_loss_weight = caffe_loss_weight
                module.caffe_propagate_down = caffe_propagate_down
                module.caffe_optimization_params = caffe_optimization_params
                for optim_param, p in zip(caffe_optimization_params, module.parameters()):
                    p.requires_grad = optim_param.get('lr_mult', 1) != 0
            else:
                print('Skipping layer [{}, {}, {}]: not found in caffemodel2pytorch.modules dict'.format(layer.name,
                                                                                                         layer_type,
                                                                                                         layer.type))

        if weights is not None:
            self.copy_from(weights)

        self.blobs = collections.defaultdict(Blob)
        self.blob_loss_weights = {name: loss_weight
                                  for module in self.children()
                                  for name, loss_weight in zip(module.caffe_output_variable_names,
                                                               module.caffe_loss_weight)}

        self.train(phase != TEST)

    def forward(self, data=None, override_variables=None, start_from=None, clear_except=None, stop_after=None,
                **variables):
        if override_variables is None:
            override_variables = dict()
        if clear_except is None:
            clear_except = dict()
        if data is not None:
            variables['data'] = data
        use_numpy = not all(map(torch.is_tensor, variables.values()))
        variables = {k: (torch.from_numpy(v.copy()) if use_numpy else v) for k, v in variables.items()}

        for module in [module for module in self.children()
                       if not all(name in variables for name in module.caffe_output_variable_names)]:
            if start_from is not None:
                if module.caffe_layer_name == start_from:
                    variables.update(override_variables)
                    start_from = None
                else:
                    continue
            # print(module.caffe_layer_name)
            for name in module.caffe_input_variable_names:
                assert name in variables, 'Variable [{}] does not exist. Pass it as a keyword argument or provide a layer which produces it.'.format(name)
            inputs = [variables[name] if propagate_down else variables[name].detach()
                      for name, propagate_down in zip(module.caffe_input_variable_names, module.caffe_propagate_down)]
            outputs = module(*inputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs, )
            if module.caffe_layer_name in clear_except:
                variables = {var_name: variables[var_name] for var_name in clear_except[module.caffe_layer_name]}
            variables.update(dict(zip(module.caffe_output_variable_names, outputs)))
            variables.update(override_variables)
            if module.caffe_layer_name == stop_after:
                break

        self.blobs.update({k: Blob(data=v, use_numpy=use_numpy) for k, v in variables.items()})
        caffe_output_variable_names = set([name for module in self.children()
                                           for name in module.caffe_output_variable_names]) - set([name for module in self.children()
                                                                                                   for name in module.caffe_input_variable_names
                                                                                                   if name not in module.caffe_output_variable_names])
        return {k: v.detach().cpu().numpy() if use_numpy else v
                for k, v in variables.items() if k in caffe_output_variable_names}

    def copy_from(self, weights):
        bytes_weights = open(weights, 'rb').read()
        bytes_parsed = self.net_param.ParseFromString(bytes_weights)
        if bytes_parsed != len(bytes_weights):
            print('caffemodel2pytorch: loading model from [{}] in caffemodel format, WARNING: file length [{}] is not equal to number of parsed bytes [{}]'.format(weights, len(bytes_weights), bytes_parsed))
        for layer in list(self.net_param.layer) + list(self.net_param.layers):
            module = getattr(self, layer.name, None)
            if module is None:
                continue
            parameter_names = []
            num_parameters = len(layer.blobs)
            if isinstance(module, BatchNormModule):
                assert num_parameters == 3, "Expected 3 parameters but got " + str(num_parameters) + " for layer: " + layer.name
                parameter_names = ['running_mean', 'running_var', 'batch_sum_multiplier']
            elif num_parameters == 1:
                parameter_names = ['weight']
            elif num_parameters == 2:
                parameter_names = ['weight', 'bias']
            else:
                assert num_parameters == 0, "Expected zero parameters but got " + str(num_parameters) + " for layer: " + layer.name

            parameters = {name: torch.FloatTensor(blob.data).view(list(blob.shape.dim) if len(blob.shape.dim) > 0
                                                                  else [blob.num, blob.channels,
                                                                        blob.height, blob.width])
                          for name, blob in zip(parameter_names, layer.blobs)}
            if len(parameters) > 0:
                module.set_parameters(**parameters)
        print('caffemodel2pytorch: loaded model from [{}] in caffemodel format'.format(weights))

    @property
    def layers(self):
        return list(self.children())


class Blob(object):
    AssignmentAdapter = type('', (object, ),
                             dict(shape=property(lambda self: self.contents.shape),
                                  __setitem__=lambda self, indices, values: setattr(self, 'contents', values)))

    def __init__(self, data=None, diff=None, use_numpy=False):
        self.data_ = data if data is not None else Blob.AssignmentAdapter()
        self.diff_ = diff if diff is not None else Blob.AssignmentAdapter()
        self.shape_ = None
        self.use_numpy = use_numpy

    def reshape(self, *args):
        self.shape_ = args

    def count(self, *axis):
        return reduce(lambda x, y: x * y, self.shape_[slice(*(axis + [-1])[:2])])

    @property
    def data(self):
        if self.use_numpy and torch.is_tensor(self.data_):
            self.data_ = self.data_.detach().cpu().numpy()
        return self.data_

    @property
    def diff(self):
        if self.use_numpy and torch.is_tensor(self.diff_):
            self.diff_ = self.diff_.detach().cpu().numpy()
        return self.diff_

    @property
    def shape(self):
        return self.shape_ if self.shape_ is not None else self.data_.shape

    @property
    def num(self):
        return self.shape[0]

    @property
    def channels(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[2]

    @property
    def width(self):
        return self.shape[3]


class Layer(torch.autograd.Function):
    def __init__(self, caffe_python_layer=None, caffe_input_variable_names=None, caffe_output_variable_names=None,
                 caffe_propagate_down=False):
        self.caffe_python_layer = caffe_python_layer
        self.caffe_input_variable_names = caffe_input_variable_names
        self.caffe_output_variable_names = caffe_output_variable_names
        self.caffe_propagate_down = caffe_propagate_down

    def forward(self, *inputs):
        bottom = [Blob(data=v.cpu().numpy()) for v in inputs]
        top = [Blob() for name in self.caffe_output_variable_names]

        #self.caffe_python_layer.reshape()
        self.caffe_python_layer.setup(bottom, top)
        self.caffe_python_layer.setup = lambda *args: None

        self.caffe_python_layer.forward(bottom, top)
        outputs = tuple(torch.from_numpy(v.data.contents.reshape(*v.shape)) for v in top)
        return outputs


modules = dict(
    Convolution=lambda param: ConvolutionModule(param),
    InnerProduct=lambda param: InnerProduct(param),
    Pooling=lambda param: PoolingModule(param),
    Softmax=lambda param: nn.Softmax(dim=param.get('axis', 1)),
    ReLU=lambda param: nn.ReLU(),
    Dropout=lambda param: nn.Dropout(p=param['dropout_ratio']),
    Eltwise=lambda param: [torch.mul, torch.add, torch.max][param.get('operation', 1)],
    LRN=lambda param: nn.LocalResponseNorm(size=param['local_size'], alpha=param['alpha'], beta=param['beta']),
    Reshape=lambda param: ReshapeModule(param),
    Scale=lambda param: ScaleModule(param),
    BatchNorm=lambda param: BatchNormModule(param),
    ProposalLayer=lambda param: ProposalLayerModule(param),
    ROIPooling=lambda param: RoIPool(output_size=(param['pooled_h'], param['pooled_w']),
                                     spatial_scale=param['spatial_scale']),
    Flatten=lambda param: torch.nn.Flatten(start_dim=param['axis'], end_dim=-1),
    ArgMax=lambda param: torch_argmax(dim=param['axis']),
    Embed=lambda param: EmbedModule(param),
    Concat=lambda param: torch_cat(dim=param['axis']),
)


def torch_argmax(dim, keepdim=False):
    return lambda input: torch.argmax(input=input, dim=dim, keepdim=keepdim)


def torch_cat(dim):
    return lambda *tensors: torch.cat(tensors=tensors, dim=dim)


class PoolingModule(nn.Module):
    def __init__(self, param):
        super(PoolingModule, self).__init__()
        self.func = [F.max_pool2d, F.avg_pool2d][param['pool']]
        self.global_pooling = param.get('global_pooling', False)
        if not self.global_pooling:
            self.kernel_size = param.get('kernel_size', 1)
            self.stride = param.get('stride', 1)
            self.padding = param.get('pad', 0)

    def forward(self, x):
        if self.global_pooling:
            # Global pooling uses kernel_size = height x width of the input
            return self.func(x, kernel_size=(x.size(-2), x.size(-1),))
        else:
            return self.func(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)


class ReshapeModule(nn.Module):
    def __init__(self, param):
        super(ReshapeModule, self).__init__()
        self.dims = param['shape']['dim']
        self.inplace = param['inplace']

    # Use existing size at idx where dim[idx] is zero
    def forward(self, x):
        shape = x.size()
        shape = [shape[idx] if self.dims[idx]==0 else self.dims[idx] for idx in range(len(self.dims))]
        if self.inplace:
            return x.reshape(shape)
        else:
            return torch.reshape(x, shape)


class ScaleModule(nn.Module):
    def __init__(self, param):
        super(ScaleModule, self).__init__()
        self.weight = nn.Parameter()
        self.bias = None if 'bias_term' in param and not param['bias_term'] else nn.Parameter()
        # self.weight_init, self.bias_init = param.get('weight_filler', {}), param.get('bias_filler', {})

    def forward(self, x):
        out = torch.mul(x, self.weight)
        if self.bias is not None:
            out += self.bias

        return out

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(self, weight=weight, bias=bias if bias.view(-1) is not None else bias)
        # Reshape the weight and bias to match input dims
        self.weight.data = self.weight.data.reshape(1, self.weight.data.size()[0], 1, 1)
        if self.bias is not None:
            self.bias.data = self.bias.data.reshape(self.weight.data.size())
        # self.in_channels = self.weight.size(1)


class BatchNormModule(nn.BatchNorm2d):
    def __init__(self, param):
        # affine=False because the caffe implementation does not include the affine layer with weight + bias
        super(BatchNormModule, self).__init__(num_features=param['num_input'], affine=False,
                                              track_running_stats=param['use_global_stats'])

    def forward(self, x):
        return super(BatchNormModule, self).forward(x)

    def set_parameters(self, running_mean, running_var, batch_sum_multiplier):
        # Should we divide or multiply here? Currently batch_sum_multiplier is always 1
        self.running_mean.data = running_mean / batch_sum_multiplier
        self.running_var.data = running_var / batch_sum_multiplier


class EmbedModule(nn.Embedding):
    def __init__(self, param):
        super(EmbedModule, self).__init__(num_embeddings=param['input_dim'], embedding_dim=param['num_output'])

    def forward(self, x):
        out = super(EmbedModule, self).forward(x)
        # Torch embeddings are in the last dim while caffe expects them in dim 1
        perm_to_caffe = tuple([0, -1] + [d for d in range(1, out.dim()-1)])
        out = out.permute(perm_to_caffe)
        return out

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(self, weight=weight, bias=bias.view(-1) if bias is not None else bias)


class FunctionModule(nn.Module):
    def __init__(self, forward):
        super(FunctionModule, self).__init__()
        self.forward_func = forward

    def forward(self, *inputs):
        return self.forward_func(*inputs)


class CaffePythonLayerModule(nn.Module):
    def __init__(self, caffe_python_layer, caffe_input_variable_names, caffe_output_variable_names, param_str):
        super(CaffePythonLayerModule, self).__init__()
        caffe_python_layer.param_str = param_str
        self.caffe_python_layer = caffe_python_layer
        self.caffe_input_variable_names = caffe_input_variable_names
        self.caffe_output_variable_names = caffe_output_variable_names

    def forward(self, *inputs):
        return Layer(self.caffe_python_layer, self.caffe_input_variable_names, self.caffe_output_variable_names)(*inputs)

    def __getattr__(self, name):
        return nn.Module.__getattr__(self, name) if name in dir(self) else getattr(self.caffe_python_layer, name)


class ConvolutionModule(nn.Module):
    def __init__(self, param):
        super(ConvolutionModule, self).__init__()

        if 'stride' in param:
            self.stride = param['stride']
        elif 'stride_h' in param:
            self.stride = (param['stride_h'], param['stride_w'])
        else:
            self.stride = 1

        if 'pad' in param:
            self.padding = param['pad']
        elif 'pad_h' in param:
            self.padding = (param['pad_h'], param['pad_w'])
        else:
            self.padding = 0

        self.dilation = param.get('dilation', 1)
        self.groups = param.get('group', 1)

        if 'kernel_size' in param:
            self.kernel_size = (param['kernel_size'], param['kernel_size'])
        elif 'kernel_h' in param:
            self.kernel_size = (param['kernel_h'], param['kernel_w'])
        else:
            self.kernel_size = (4, 4)  # caffe default for kernel_size is 4

        self.weight = None
        self.bias = param.get('bias_term', True)
        self.weight = nn.Parameter()
        self.bias = None if 'bias_term' in param and not param['bias_term'] else nn.Parameter()
        self.weight_init, self.bias_init = param.get('weight_filler', {}), param.get('bias_filler', {})

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(self, weight=weight, bias=bias.view(-1) if bias is not None else bias)
        self.in_channels = self.weight.size(1)


class InnerProduct(nn.Linear):
    def __init__(self, param):
        super(InnerProduct, self).__init__(1, param['num_output'])
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get('weight_filler', {}), param.get('bias_filler', {})

    def forward(self, x):
        if self.weight.numel() == 0 and self.bias.numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad]
            super(InnerProduct, self).__init__(x.size(1), self.out_features)
            init_weight_bias(self, requires_grad=requires_grad)

        # Caffe expects [N, C, *] while Torch works with [N, *, C]
        perm_to_pytorch = tuple([0] + [d for d in range(2, x.dim())] + [1])
        perm_reverse = tuple([0, -1] + [d for d in range(1, x.dim()-1)])
        out = super(InnerProduct, self).forward(x.permute(perm_to_pytorch))
        out = out.permute(perm_reverse)
        return out

    def set_parameters(self, weight=None, bias=None):
        # init_weight_bias(self, weight=weight.view(weight.size(-2), weight.size(-1)) if weight is not None else None,
        #                  bias=bias.view(-1) if bias is not None else None)
        init_weight_bias(self, weight=weight if weight is not None else None,
                         bias=bias.view(-1) if bias is not None else None)
        self.in_features = self.weight.size(1)


def init_weight_bias(self, weight=None, bias=None, requires_grad=[]):
    if weight is not None:
        self.weight = nn.Parameter(weight.type_as(self.weight), requires_grad=self.weight.requires_grad)
    if bias is not None:
        self.bias = nn.Parameter(bias.type_as(self.bias), requires_grad=self.bias.requires_grad)
    for name, requires_grad in zip(['weight', 'bias'], requires_grad):
        param, init = getattr(self, name), getattr(self, name + '_init')
        if init.get('type') == 'gaussian':
            nn.init.normal_(param, std = init['std'])
        elif init.get('type') == 'constant':
            nn.init.constant_(param, val = init['value'])
        param.requires_grad = requires_grad


def first_or(param, key, default):
    return param[key] if isinstance(param.get(key), int) else (param.get(key, []) + [default])[0]


def to_dict(obj):
    return list(map(to_dict, obj)) if isinstance(obj, collections.Iterable) \
        else {} if obj is None \
        else {f.name: converter(v) if f.label != FD.LABEL_REPEATED else list(map(converter, v))
              for f, v in obj.ListFields()
              for converter in [{FD.TYPE_DOUBLE: float, FD.TYPE_SFIXED32: float, FD.TYPE_SFIXED64: float,
                                 FD.TYPE_SINT32: int, FD.TYPE_SINT64: int, FD.TYPE_FLOAT: float, FD.TYPE_ENUM: int,
                                 FD.TYPE_UINT32: int, FD.TYPE_INT64: int, FD.TYPE_UINT64: int, FD.TYPE_INT32: int,
                                 FD.TYPE_FIXED64: float, FD.TYPE_FIXED32: float, FD.TYPE_BOOL: bool,
                                 FD.TYPE_STRING: str, FD.TYPE_BYTES: lambda x: x.encode('string_escape'),
                                 FD.TYPE_MESSAGE: to_dict}[f.type]]}
