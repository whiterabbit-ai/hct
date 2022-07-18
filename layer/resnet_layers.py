# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import torch.nn as nn
from torchvision.models.resnet import conv3x3


class BasicBlockV2(nn.Module):
    """Basic Residual Block of ResNet V2.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        # # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class ResNetV2(nn.Module):
    """Adapted fom torchvision ResNet, converted to v2."""

    def __init__(
        self,
        input_channels,
        num_filters,
        first_layer_kernel_size,
        first_layer_conv_stride,
        blocks_per_layer_list,
        block_strides_list,
        block_fn,
        first_layer_padding=0,
        first_pool_size=None,
        first_pool_stride=None,
        first_pool_padding=0,
        num_chns=[],
    ):
        super().__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )

        '''
        The warning due to this MaxPooling is PyTorch related.
        According to `[1]`_, it should be fixed soon.
        .. _[1] https://discuss.pytorch.org/t/named-tensors-warning/130019/3
        '''
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters

        for i, (num_blocks, stride, num_filters) in enumerate(
            zip(blocks_per_layer_list, block_strides_list, num_chns)
        ):
            current_num_filters = num_filters
            self.layer_list.append(
                self._make_layer(
                    block=block_fn,
                    planes=current_num_filters,
                    blocks=num_blocks,
                    stride=stride,
                )
            )

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.num_chns = num_chns

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)

        for _, layer in enumerate(self.layer_list):
            h = layer(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):

        # GMIC downsampling Style
        downsample = nn.Sequential(
            nn.Conv2d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
        )
        layers_ = [block(self.inplanes, planes, stride, downsample)]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)
