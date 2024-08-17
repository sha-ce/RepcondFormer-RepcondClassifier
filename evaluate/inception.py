
# copied https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py and changed something for signal

import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

FID_WEIGHTS_URL='./classify_signal_inception_weight.pth'

class InceptionForSignal(nn.Module):
    """Pretrained Inception network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        num_classes: int = 1000,
        resize_input: bool = True,
        aux_logits: bool = True,
        init_weights: Optional[bool] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.resize_input = resize_input
        self.aux_logits = aux_logits
        self.Conv1d_1a_3 = BasicConv1d(3, 32, kernel_size=3, stride=2)
        self.Conv1d_2a_3 = BasicConv1d(32, 32, kernel_size=3)
        self.Conv1d_2b_3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.Conv1d_3b_1 = BasicConv1d(64, 80, kernel_size=1)
        self.Conv1d_4a_3 = BasicConv1d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7=128)
        self.Mixed_6c = InceptionC(768, channels_7=160)
        self.Mixed_6d = InceptionC(768, channels_7=160)
        self.Mixed_6e = InceptionC(768, channels_7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048, is_maxpool=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # for calculated fid
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        block0 = [
            self.Conv1d_1a_3,
            self.Conv1d_2a_3,
            self.Conv1d_2b_3,
            self.maxpool1,
        ]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [
                self.Conv1d_3b_1,
                self.Conv1d_4a_3,
                self.maxpool2,
            ]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [
                self.Mixed_5b,
                self.Mixed_5c,
                self.Mixed_5d,
                self.Mixed_6a,
                self.Mixed_6b,
                self.Mixed_6c,
                self.Mixed_6d,
                self.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [
                self.Mixed_7a,
                self.Mixed_7b,
                self.Mixed_7c,
                self.avgpool,
            ]
            self.blocks.append(nn.Sequential(*block3))

    def _resized_input(self, x: Tensor) -> Tensor:
        if self.resize_input:
            x = F.interpolate(x, size=299, mode='linear', align_corners=False)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # x.shape: ([N, 3, 299])
        x = self.Conv1d_1a_3(x)
        # x.shape: ([N, 32, 149])
        x = self.Conv1d_2a_3(x)
        # x.shape: ([N, 32, 147])
        x = self.Conv1d_2b_3(x)
        # x.shape: ([N, 64, 147])
        x = self.maxpool1(x)
        # x.shape: ([N, 64, 73])
        x = self.Conv1d_3b_1(x)
        # x.shape: ([N, 80, 73])
        x = self.Conv1d_4a_3(x)
        # x.shape: ([N, 192, 71])
        x = self.maxpool2(x)
        # x.shape: ([N, 192, 35])
        x = self.Mixed_5b(x)
        # x.shape: ([N, 256, 35])
        x = self.Mixed_5c(x)
        # x.shape: ([N, 288, 35])
        x = self.Mixed_5d(x)
        # x.shape: ([N, 288, 35])
        x = self.Mixed_6a(x)
        # x.shape: ([N, 768, 17])
        x = self.Mixed_6b(x)
        # x.shape: ([N, 768, 17])
        x = self.Mixed_6c(x)
        # x.shape: ([N, 768, 17])
        x = self.Mixed_6d(x)
        # x.shape: ([N, 768, 17])
        x = self.Mixed_6e(x)
        # x.shape: ([N, 768, 17])
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # x.shape: ([N, 768, 17])
        x = self.Mixed_7a(x)
        # x.shape: ([N, 1280, 8])
        x = self.Mixed_7b(x)
        # x.shape: ([N, 2048, 8])
        x = self.Mixed_7c(x)
        # x.shape: ([N, 2048, 8])
        # Adaptive average pooling
        x = self.avgpool(x)
        # x.shape: ([N, 2048, 1])
        x = self.dropout(x)
        # x.shape: ([N, 2048, 1])
        x = torch.flatten(x, 1)
        # x.shape: ([N, 2048])
        x = self.fc(x)
        # x.shape: ([N, 1000(num_classes)])
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._resized_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)
    
    def from_pretrained(self, path=FID_WEIGHTS_URL):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)
        return self
    
    def forward_eval(self, x):
        output = []
        x = self._resized_input(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                output.append(x)
            if idx == self.last_needed_block:
                break
        return output[0], self.fc(torch.flatten(self.dropout(output[0]), 1))


class InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        self.branch1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch5_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch5_2 = BasicConv1d(48, 64, kernel_size=5, padding=2)
        self.branch3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3dbl_3 = BasicConv1d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv1d(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)

        branch5 = self.branch5_1(x)
        branch5 = self.branch5_2(branch5)

        branch3dbl = self.branch3dbl_1(x)
        branch3dbl = self.branch3dbl_2(branch3dbl)
        branch3dbl = self.branch3dbl_3(branch3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1, branch5, branch3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3 = BasicConv1d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3dbl_3 = BasicConv1d(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3 = self.branch3(x)

        branch3dbl = self.branch3dbl_1(x)
        branch3dbl = self.branch3dbl_2(branch3dbl)
        branch3dbl = self.branch3dbl_3(branch3dbl)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)

        outputs = [branch3, branch3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7: int) -> None:
        super().__init__()
        self.branch1 = BasicConv1d(in_channels, 192, kernel_size=1)
        c7 = channels_7
        self.branch7_1 = BasicConv1d(in_channels, c7, kernel_size=1)
        self.branch7_2 = BasicConv1d(c7, 192, kernel_size=7, padding=3)
        self.branch7dbl_1 = BasicConv1d(in_channels, c7, kernel_size=1)
        self.branch7dbl_2 = BasicConv1d(c7, c7, kernel_size=7, padding=3)
        self.branch7dbl_3 = BasicConv1d(c7, 192, kernel_size=7, padding=3)
        self.branch_pool = BasicConv1d(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)

        branch7 = self.branch7_1(x)
        branch7 = self.branch7_2(branch7)

        branch7dbl = self.branch7dbl_1(x)
        branch7dbl = self.branch7dbl_2(branch7dbl)
        branch7dbl = self.branch7dbl_3(branch7dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1, branch7, branch7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3_1 = BasicConv1d(in_channels, 192, kernel_size=1)
        self.branch3_2 = BasicConv1d(192, 320, kernel_size=3, stride=2)
        self.branch7x3_1 = BasicConv1d(in_channels, 192, kernel_size=1)
        self.branch7x3_2 = BasicConv1d(192, 192, kernel_size=7, padding=3)
        self.branch7x3_3 = BasicConv1d(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)

        branch7x3 = self.branch7x3_1(x)
        branch7x3 = self.branch7x3_2(branch7x3)
        branch7x3 = self.branch7x3_3(branch7x3)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)
        outputs = [branch3, branch7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int, is_maxpool: bool=False) -> None:
        super().__init__()
        self.is_maxpool = is_maxpool
        self.branch1 = BasicConv1d(in_channels, 320, kernel_size=1)
        self.branch3_1 = BasicConv1d(in_channels, 384, kernel_size=1)
        self.branch3_2 = BasicConv1d(384, 768, kernel_size=3, padding=1)
        self.branch3dbl_1 = BasicConv1d(in_channels, 448, kernel_size=1)
        self.branch3dbl_2 = BasicConv1d(448, 384, kernel_size=3, padding=1)
        self.branch3dbl_3 = BasicConv1d(384, 768, kernel_size=3, padding=1)
        self.branch_pool = BasicConv1d(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)

        branch3dbl = self.branch3dbl_1(x)
        branch3dbl = self.branch3dbl_2(branch3dbl)
        branch3dbl = self.branch3dbl_3(branch3dbl)

        if self.is_maxpool:
            branch_pool = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        else:
            branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1, branch3, branch3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv0 = BasicConv1d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv1d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool1d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        # Adaptive average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


