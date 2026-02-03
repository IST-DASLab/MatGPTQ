import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["FakeMatryoshkaLinear"]

def matryoshka_slice(x: Tensor, master_bitwidth: int, slice_bitwidth: int):    
    drop = 8 - slice_bitwidth # always assume 8 bit store
    floor = x >> drop
    
    if master_bitwidth == slice_bitwidth:
        return torch.clamp(floor, 0, 2**slice_bitwidth - 1)
    
    round_bit = (x >> (drop - 1)) & 1
    return torch.clamp(floor + round_bit, 0, 2**slice_bitwidth - 1)

def dequantize(q, scale, zero, master_bitwidth, slice_bitwidth):
    q_slices = matryoshka_slice(q, master_bitwidth, slice_bitwidth)
    q_shifted = q_slices << (master_bitwidth - slice_bitwidth)
    return scale * (q_shifted - zero)

def quantize(x, scale, zero, maxq, bit_options, bit_weights):
    master_bitwidth = max(bit_options)
    i = torch.arange(maxq + 1, device=x.device, dtype=torch.uint8)

    shape = [maxq + 1] + [1] * x.dim()
    q = i.view(shape).expand(-1, *x.shape)

    x = x.unsqueeze(0)  # add batch dimension for broadcasting

    # loop unrolled for performance
    total_err = 0
    loss = 0
    if 8 in bit_options:
        l = (x - dequantize(q, scale, zero, master_bitwidth, master_bitwidth))
        loss += l
        total_err += bit_weights.get(8, 1.0) * (l ** 2)
    if 7 in bit_options:
        l = (x - dequantize(q, scale, zero, master_bitwidth, 7))
        loss += l
        total_err += bit_weights.get(7, 1.0) * (l ** 2)
    if 6 in bit_options:
        l = (x - dequantize(q, scale, zero, master_bitwidth, 6))
        loss += l
        total_err += bit_weights.get(6, 1.0) * (l ** 2)
    if 5 in bit_options:
        l = (x - dequantize(q, scale, zero, master_bitwidth, 5))
        loss += l
        total_err += bit_weights.get(5, 1.0) * (l ** 2)
    if 4 in bit_options:
        l = (x - dequantize(q, scale, zero, master_bitwidth, 4))
        loss += l
        total_err += bit_weights.get(4, 1.0) * (l ** 2)
    if 3 in bit_options:
        l = (x - dequantize(q, scale, zero, master_bitwidth, 3))
        loss += l
        total_err += bit_weights.get(3, 1.0) * (l ** 2)
    if 2 in bit_options:
        l = (x - dequantize(q, scale, zero, master_bitwidth, 2))
        loss += l
        total_err += bit_weights.get(2, 1.0) * (l ** 2)

    qweight = torch.argmin(total_err, dim=0)
    return qweight.to(torch.uint8), loss.gather(0, qweight.unsqueeze(0)).squeeze(0)

class MatryoshkaQuantizer(nn.Module):
    def __init__(self, shape=1):
        super(MatryoshkaQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        norm=2.0,
        grid=100,
        maxshrink=0.8,
        reserved_bins: int = 0,
    ):
        self.bits = bits
        self.maxq = torch.tensor(2**bits - 1 - reserved_bins)
        self.perchannel = perchannel
        self.sym = sym
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        xmin = x.min(1).values
        xmax = x.max(1).values

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = xmin == xmax
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x, bitwidth_options, bitwidth_weights=None):
        if self.ready():
            return quantize(
                x, self.scale, self.zero, self.maxq, bitwidth_options, bitwidth_weights
            )
        return x

    def dequantize(self, x, master_bitwidth, slice_bitwidth):
        if self.ready():
            return dequantize(
                x, self.scale, self.zero, master_bitwidth, slice_bitwidth
            )
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class FakeMatryoshkaLinear(nn.Module):

    def __init__(
        self, qweight, scale, zero, bias=None, perm=None, master_bitwidth=8, slice_bitwidth=8
    ) -> None:
        assert (
            slice_bitwidth <= master_bitwidth
        ), "slice_bits must be less than or equal to max_bits"
        super().__init__()
        self.master_bitwidth = master_bitwidth
        self.slice_bitwidth = slice_bitwidth
        self.in_features = qweight.shape[1]
        self.out_features = qweight.shape[0]
        self.perm = perm
        if perm is not None:
            self.invperm = perm.argsort()
        else:
            self.invperm = None

        self.register_buffer("qweight", qweight)
        self.scale = nn.Parameter(scale)
        self.register_buffer("zero", zero)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def get_weight(self):
        qweight = self.qweight

        scale = self.scale.view(qweight.shape[0], -1, 1)
        zero = self.zero.view(qweight.shape[0], -1, 1)

        num_groups = scale.shape[1]
        weight = dequantize(
            qweight.view(qweight.shape[0], num_groups, -1),
            scale,
            zero,
            self.master_bitwidth,
            self.slice_bitwidth,
        ).view_as(qweight)
        return weight

    def forward(self, input: torch.Tensor):
        if self.perm is not None:
            input = input[..., self.perm]
        # get weight without outliers
        weight = self.get_weight()
        out = F.linear(input, weight, self.bias)
        return out
