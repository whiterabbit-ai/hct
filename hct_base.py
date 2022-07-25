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

import dill
import torch
import torch.nn as nn
from layer import ac_layers
from functools import partial
from layer import resnet_layers


class HCTBase(nn.Module):
    """Create HCT Model following `[1]`_.

    The model has `num_conv_stages` convolutional stages followed by
    (5-`num_conv_stages`) Attention-Convolution (AC) stages.

    _[1] TODO add arXiv Link

    Args:
        num_classes (int, optional): Number of classification logits. Defaults to 2.
        pretrained_weights (str, optional): Path to load pretrained weights.
            Defaults to None.
        input_chns (int, optional): Number of input channels. Defaults to 1.
        first_layer_stride (int, optional): Stride inside the first ResNet Stage.
            By setting this stride to 2, the model becomes more compact. Defaults to 2.
        attention_dropout_rate (float, optional): The dropout rate in the self-attention module.
                Defaults to 0.1.
        use_performer_RELU (bool, optional): Use performer RELU. Defaults to True.
            When false, Performer Softmax is used.
        num_conv_stages (int, optional): The number of conv stages in GMIC. Defaults to 3.
            The recommended range [2,4]. When num_conv_stages ==5, HCT turns into GMIC.
        blocks_per_stage (list, optional): Number of blocks per stage.
            Defaults to [2, 2, 2, 2, 2].

    Attributes:
        _num_conv_stages (int): Number of vanilla convolutional stages.
        conv_stages (ResNetV2): A ResNetV2 network with `num_conv_stages` stages.
        ac_stages (ACNet): A ACNet network with (5-`num_conv_stages`) stages.
        final_bn (nn.BatchNorm2d): BatchNorm applied on the network's learned representation before
            applying the linear classifier.
        relu (nn.Relu): A non-linear activation applied on the network's learned representation.
        classifier (nn.Linear): A linear classifier applied on network's learned representation,
            i.e., classifier(relu(final_bn(f))) where f is the network's learned representation.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained_weights=None,
        input_chns=1,
        first_layer_stride=2,
        attention_dropout_rate=0.1,
        use_performer_RELU=True,
        num_conv_stages=3,
        blocks_per_stage=[2, 2, 2, 2, 2],
    ):
        super().__init__()
        num_chns_per_stage = [16, 32, 64, 128, 256]

        conv_block_num_chns = num_chns_per_stage[:num_conv_stages]
        ac_block_num_chns = num_chns_per_stage[num_conv_stages:]

        conv_blocks_per_stage = blocks_per_stage[:num_conv_stages]
        ac_blocks_per_stage = blocks_per_stage[num_conv_stages:]

        strides_per_stage = [first_layer_stride, 2, 2, 2, 2]
        conv_strides_per_stage = strides_per_stage[:num_conv_stages]
        ac_strides_per_stage = strides_per_stage[num_conv_stages:]

        self.conv_stages = resnet_layers.ResNetV2(
            input_channels=input_chns,
            num_filters=16,
            # first conv layer
            first_layer_kernel_size=(7, 7),
            first_layer_conv_stride=2,
            first_layer_padding=3,
            # first pooling layer
            first_pool_size=3,
            first_pool_stride=2,
            first_pool_padding=0,
            # res blocks architecture
            blocks_per_layer_list=conv_blocks_per_stage,
            block_strides_list=conv_strides_per_stage,
            block_fn=resnet_layers.BasicBlockV2,
            num_chns=conv_block_num_chns,
        )
        current_num_channels = conv_block_num_chns[-1]
        if num_conv_stages < 5:
            self.ac_stages = ac_layers.ACNet(
                input_channels=current_num_channels,
                blocks_per_layer_list=ac_blocks_per_stage,
                block_strides_list=ac_strides_per_stage,
                block_fn=partial(
                    ac_layers.ACBlock,
                    dropout_rate=attention_dropout_rate,
                    generalized_attention=use_performer_RELU,
                ),
                num_chns=ac_block_num_chns,
            )
            current_num_channels = ac_block_num_chns[-1]
        else:
            self.ac_stages = None
            current_num_channels = conv_block_num_chns[-1]

        self._num_conv_stages = num_conv_stages
        self.classifier = nn.Linear(current_num_channels, num_classes)

        self.relu = nn.ReLU()
        self.final_bn = nn.BatchNorm2d(current_num_channels)

        # initialize weights
        self.init_weights(pretrained_weights)

    def init_weights(self, pretrained_weights):
        """Initialize HCT model from either pre-trained weights or kaiming normal.

        Args:
            pretrained_weights (str): Path to pre-trained weights saved using torch.save.
                If pretrained_weights=None, kaiming normal is used.
        """
        if pretrained_weights is not None:
            with open(pretrained_weights, 'rb') as f:
                checkpoint = torch.load(
                    f, pickle_module=dill, map_location=lambda storage, loc: storage
                )
                self.load_state_dict(checkpoint, strict=True)
                return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        """HCT feedforward method. HCT goes through vanilla convolutional stages before going
            through the Attention-Conv (AC) stages.

        Args:
            x (Torch.Tensor): An input tensor of size [bz,chns,rows,cols],
                where bz is the batch-size and chns is the number of input channels.

        Returns:
            Torch.Tensor: HCT output after feedforward. The output tensor shape will be
                [bz, num_classes].
        """

        x = self.conv_stages(x)
        if self._num_conv_stages < 5:
            x = self.ac_stages(x)

        x = self.final_bn(x)
        x = self.relu(x)
        x = x.mean([2, 3])  # global average pool
        x = self.classifier(x)

        return x


def main():
    # Toy HCT forward run
    # bz, ch, rows, cols = 32, 1, 3328, 2560  # Fits on 48GB GPUs
    bz, ch, rows, cols = 32, 1, 512, 512
    input_data = torch.rand((bz, ch, rows, cols))
    hct = HCTBase()
    model_output = hct(input_data)
    print(model_output.shape)


if __name__ == '__main__':
    main()
