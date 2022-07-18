# Copyright 2022 WhiteRabbit.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from torchvision.models.resnet import conv3x3
from layer.performer_attention import PerformerAttention


class ACBlock(nn.Module):
    """Attention-Convolution (AC) Block.

        This block replaces the first conv layer in BasicBlock with an attention layer with
        linear complexity.

    Args:
        d_in (int): Number of input channels.
        d_out (int): Number of output channels.
        dropout_rate (float, optional): The dropout rate used by Performers. Defaults to 0.1.
        generalized_attention (bool, optional): Whether to use RELU Kernel or Softmax kernel for
            attention. True denotes RELU while False denotes Softmax. Defaults to True.
        stride (int, optional): The stride used by the convolutional layer. Defaults to 1.
        downsample (nn.Sequential, optional): The downsampling module used within the
            ResNet-identify path when stride=2. Defaults to None.
        n_head (int, optional): Number of attention heads in Performers. Defaults to 8.

    Attributes:
        d_k (int): Keys-dimension
        d_q (int): Query-dimension
        d_v (int): Value-dimension
        n_head (int): Number of self-attention heads
        performer_attention (nn.Module): A self-attention Performer module with linear complexity.
        downsample (nn.Module): The downsampling module used within the
            ResNet-identify path when stride=2.
        conv2 (nn.Conv2d): Conv layer responsible for both downsampling and increasing the number 
            of dimensions.
        bn1 (nn.BatchNorm2d): BatchNorm layer applied before self-attention layer (block input).
        bn1 (nn.BatchNorm2d): BatchNorm layer applied after the self-attention layer 
            (self-attention output).
        relu (nn.ReLU): activation function applied throughout the AC block.
    """
    expansion = 1
    def __init__(
        self,
        d_in,
        d_out,
        dropout_rate=0.1,
        generalized_attention=True,
        stride=1,
        downsample=None,
        n_head=8,
    ):
        super().__init__()
        assert d_in % n_head == 0, 'The input hidden-dimension should be divisible by # of heads'

        self.d_k = self.d_q = self.d_v = d_in // n_head
        self.n_head = n_head

        nb_features = None
        kernel_fn = nn.ReLU()
        qkv_bias = True
        attn_out_bias = True
        dim_head = d_in // n_head

        self.performer_attention = PerformerAttention(
            dim=d_in,
            heads=n_head,
            dim_head=dim_head,
            nb_features=nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            qkv_bias=qkv_bias,
            attn_out_bias=attn_out_bias,
            dropout_rate=dropout_rate,
        )

        self.downsample = downsample
        # This conv layer is responsible for both downsampling the resolution and
        # increasing the number of channels
        self.conv2 = conv3x3(d_in, d_out, stride=stride)
        self.bn1 = nn.BatchNorm2d(d_in)
        self.bn2 = nn.BatchNorm2d(d_in)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Feedforward method for AC block.

        Args:
            x (torch.tensor): Input torch tensor.

        Returns:
            torch.tensor: Output torch tensor.
        """
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:  ## MICCAI HCT
            residual = self.downsample(out)
        # Self-attention using performers
        out = self.performer_attention(out)
        out += x  ## residual with input x

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class ACNet(nn.Module):
    """ACNet contains multiple Attention-Convolutional (AC) stages. 

        The ACNet contains a list of AC-stages. Each stage is composed of multiple AC-blocks.

    Args:
        input_channels (int): Number of input channels.
        block_fn (nn.Module): The block module used inside each AC stage.
        blocks_per_layer_list (list[int]): Number of blocks with each stage.
            The list length denotes the number of stages.
        block_strides_list (list[int]): the stride within the first block of each stage.
            The list length denotes the number of stages.
        num_chns (list[int]): The number of output channels after each AC-stage.

    Attributes:
        layer_list (ModuleList): A list of AC stages with the AC-Net.
        inplanes (int): Number of channels (filters) in the input feature-map.
        num_chns (list[int]): The number of output channels after each AC-stage.
    """
    def __init__(
        self,
        input_channels,
        block_fn,
        blocks_per_layer_list,
        block_strides_list,
        num_chns,
    ):
        
        super().__init__()
        self.layer_list = nn.ModuleList()
        self.inplanes = input_channels
        for i, (num_blocks, stride, num_filters) in enumerate(
            zip(blocks_per_layer_list, block_strides_list, num_chns)
        ):

            self.layer_list.append(
                self._make_layer(
                    block=block_fn,
                    planes=num_filters,
                    num_blocks=num_blocks,
                    stride=stride,
                )
            )

        self.num_chns = num_chns

    def forward(self, x):
        """Feedforward through multiple AC stages.

        Args:
            x (torch.tensor): Input torch tensor.

        Returns:
            torch.tensor: Output torch tensor.
        """
        h = x
        for layer in self.layer_list:
            h = layer(h)
        return h

    def _make_layer(self, block, planes, num_blocks, stride=1):
        """Make an AC stage with `num_blocks` blocks.

        Args:
            block (nn.Module): The block module used to create the AC stage.
            planes (int): Number of output channels in each block.
            num_blocks (int): Number of blocks with an AC stage.
            stride (int, optional): The stride within the first block. Defaults to 1.

        Returns:
            nn.Module: The created AC stage.
        """

        # GMIC downsampling Style
        downsample = nn.Sequential(
            nn.Conv2d(
                self.inplanes,
                planes * block.func.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
        )

        layers_ = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        ]

        self.inplanes = planes * block.func.expansion
        for i in range(1, num_blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)
