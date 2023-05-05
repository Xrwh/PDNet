import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" PDNet Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        # x = shortcut + x
        return x


class PDNet(nn.Module):
    r""" PDNet
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
                             # nn.BatchNorm2d(dims[0], eps=1e-6))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
            # downsample_layer = nn.Sequential(nn.BatchNorm2d(dims[i], eps=1e-6),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.norm = nn.BatchNorm2d(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)


        # x = self.norm(x)
        # x = x.mean([-2, -1])
        # return x
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def PDNet_tiny(num_classes: int):
    model = PDNet(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def PDNet_small(num_classes: int):
    model = PDNet(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def PDNet_base(num_classes: int):
    model = PDNet(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def PDNet_large(num_classes: int):
    model = PDNet(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def PDNet_xlarge(num_classes: int):
    model = PDNet(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model

# from timm.models.helpers import build_model_with_cfg
# from layers import Sequencer2DBlock, PatchEmbed, LSTM2D, GRU2D, RNN2D, Downsample2D
# from timm.models.layers import lecun_normal_, Mlp
# from functools import partial
#
# def get_stage(index, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer, rnn_layer, mlp_layer,
#               norm_layer, act_layer, num_layers, bidirectional, union,
#               with_fc, drop=0., drop_path_rate=0., **kwargs):
#     assert len(layers) == len(patch_sizes) == len(embed_dims) == len(hidden_sizes) == len(mlp_ratios)
#     blocks = []
#     for block_idx in range(layers[index]):
#         drop_path = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
#         blocks.append(block_layer(embed_dims[index], hidden_sizes[index], mlp_ratio=mlp_ratios[index],
#                                   rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer,
#                                   act_layer=act_layer, num_layers=num_layers,
#                                   bidirectional=bidirectional, union=union, with_fc=with_fc,
#                                   drop=drop, drop_path=drop_path))
#
#     if index < len(embed_dims) - 1:
#         blocks.append(Downsample2D(embed_dims[index], embed_dims[index + 1], patch_sizes[index + 1]))
#
#     blocks = nn.Sequential(*blocks)
#     return blocks
#
# class Sequencer2D(nn.Module):
#     def __init__(
#             self,
#             num_classes=2,
#             img_size=224,
#             in_chans=3,
#             layers=[4, 3, 14, 3],
#             patch_sizes=[7, 2, 1, 1],
#             embed_dims=[192, 384, 384, 384],
#             hidden_sizes=[48, 96, 96, 96],
#             mlp_ratios=[3.0, 3.0, 3.0, 3.0],
#             block_layer=Sequencer2DBlock,
#             rnn_layer=LSTM2D,
#             mlp_layer=Mlp,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             act_layer=nn.GELU,
#             num_rnn_layers=1,
#             bidirectional=True,
#             union="cat",
#             with_fc=True,
#             drop_rate=0.,
#             drop_path_rate=0.,
#             nlhb=False,
#             stem_norm=False,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = embed_dims[0]  # num_features for consistency with other models
#         self.embed_dims = embed_dims
#         self.stem = PatchEmbed(
#             img_size=img_size, patch_size=patch_sizes[0], in_chans=in_chans,
#             embed_dim=embed_dims[0], norm_layer=norm_layer if stem_norm else None,
#             flatten=False)
#
#         self.blocks = nn.Sequential(*[
#             get_stage(
#                 i, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer=block_layer,
#                 rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer,
#                 num_layers=num_rnn_layers, bidirectional=bidirectional,
#                 union=union, with_fc=with_fc, drop=drop_rate, drop_path_rate=drop_path_rate,
#             )
#             for i, _ in enumerate(embed_dims)])
#
#         self.norm = norm_layer(embed_dims[-1])
#         self.head = nn.Linear(embed_dims[-1], self.num_classes) if num_classes > 0 else nn.Identity()
#
#         self.init_weights(nlhb=nlhb)
#
#     # def init_weights(self, nlhb=False):
#     #     head_bias = -math.log(self.num_classes) if nlhb else 0.
#     #     named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward_features(self, x):
#         x = self.stem(x)
#         x = self.blocks(x)
#         x = self.norm(x)
#         x = x.mean(dim=(1, 2))
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x
# #
# # def _create_sequencer2d(variant, pretrained=False, **kwargs):
# #     if kwargs.get('features_only', None):
# #         raise RuntimeError('features_only not implemented for Sequencer2D models.')
# #
# #     model = build_model_with_cfg(
# #         Sequencer2D, variant, pretrained,
# #         default_cfg=default_cfgs[variant],
# #         pretrained_filter_fn=checkpoint_filter_fn,
# #         **kwargs)
# #     return model
#
# # def sequencer2d_m(pretrained=False, **kwargs):
# #     model_args = dict(
# #         layers=[4, 3, 14, 3],
# #         patch_sizes=[7, 2, 1, 1],
# #         embed_dims=[192, 384, 384, 384],
# #         hidden_sizes=[48, 96, 96, 96],
# #         mlp_ratios=[3.0, 3.0, 3.0, 3.0],
# #         rnn_layer=LSTM2D,
# #         bidirectional=True,
# #         union="cat",
# #         with_fc=True,
# #         **kwargs)
# #     model = _create_sequencer2d('sequencer2d_m', pretrained=pretrained, **model_args)
# #     return model
# import einops
# #
# class LSTMnet(nn.Module):
#     def __init__(self, input_size, hidden_size=256, n_class=2):
#         super(LSTMnet, self).__init__()
#
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=4)
#         self.fc = nn.Linear(hidden_size, n_class)
#
#     def forward(self, x): # x‘s shape (batch_size, 序列长度, 序列中每个数据的长度)
#         x = einops.rearrange(x,'b c h w -> h b (c w)')
#         output, (h, c) = self.lstm(x)
#         out = self.fc(output[-1])
#
#         return out
