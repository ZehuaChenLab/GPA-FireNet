import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient channel attention used by ELDA pre-enhancement."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1).sigmoid()
        return x * weights



class ESA(nn.Module):
    """Efficient spatial attention with a bounded learnable mixture."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.alpha_logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.pool(x)
        maximum = centered.amax(dim=1, keepdim=True).sigmoid()
        minimum = centered.amin(dim=1, keepdim=True).sigmoid()
        alpha = self.alpha_logit.sigmoid()
        return x * (alpha * maximum + (1.0 - alpha) * minimum)



class EAA(nn.Module):
    """Scaled dot-product attention in full-axis strips.

    A horizontal strip spans the complete height and ``window_size`` columns;
    a vertical strip spans ``window_size`` rows and the complete width. This
    follows the surviving ELDA implementation while adding safe padding and
    the missing 1/sqrt(d_k) scaling. Its attention-map complexity is
    O(H*W*s*(H+W)), lower than full 2-D attention but not strictly linear.
    """

    def __init__(
        self,
        channels: int,
        axis: str,
        num_heads: int = 8,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        if axis not in {"horizontal", "vertical"}:
            raise ValueError("axis must be 'horizontal' or 'vertical'")
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.axis = axis
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

    def _to_windows(self, x: torch.Tensor):
        batch, channels, height, width = x.shape
        size = self.window_size
        if self.axis == "horizontal":
            pad = (-width) % size
            x = F.pad(x, (0, pad, 0, 0))
            padded_height, padded_width = height, width + pad
            count = padded_width // size
            x = x.reshape(batch, channels, height, count, size)
            x = x.permute(0, 3, 2, 4, 1).reshape(
                batch * count, height * size, channels
            )
        else:
            pad = (-height) % size
            x = F.pad(x, (0, 0, 0, pad))
            padded_height, padded_width = height + pad, width
            count = padded_height // size
            x = x.reshape(batch, channels, count, size, width)
            x = x.permute(0, 2, 3, 4, 1).reshape(
                batch * count, size * width, channels
            )

        tokens = x.shape[1]
        x = x.reshape(-1, tokens, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        restore = (
            batch,
            channels,
            height,
            width,
            padded_height,
            padded_width,
            count,
        )
        return x, restore

    def _from_windows(self, x: torch.Tensor, restore) -> torch.Tensor:
        batch, channels, height, width, padded_height, padded_width, count = restore
        size = self.window_size
        x = x.permute(0, 2, 1, 3).contiguous()
        if self.axis == "horizontal":
            x = x.reshape(batch, count, padded_height, size, channels)
            x = x.permute(0, 4, 2, 1, 3).reshape(
                batch, channels, padded_height, padded_width
            )
            return x[:, :, :height, :width].contiguous()
        x = x.reshape(batch, count, size, padded_width, channels)
        x = x.permute(0, 4, 1, 2, 3).reshape(
            batch, channels, padded_height, padded_width
        )
        return x[:, :, :height, :width].contiguous()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q_windows, restore = self._to_windows(q)
        k_windows, _ = self._to_windows(k)
        v_windows, _ = self._to_windows(v)
        attention = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = attention @ v_windows
        return self._from_windows(output, restore)



class ELDA(nn.Module):
    """Efficient Linear Dual-Axial Attention from the manuscript."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        hidden = int(channels * mlp_ratio)
        self.eca = ECA()
        self.esa = ESA()
        self.position1 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.input_projection = nn.Conv2d(channels, channels, 1)
        self.input_depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.qkv = nn.Conv2d(channels, 5 * channels, 1)
        self.horizontal = EAA(
            channels, "horizontal", num_heads, window_size
        )
        self.vertical = EAA(
            channels, "vertical", num_heads, window_size
        )
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.position2 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.eca(x) + self.esa(x)
        positioned = enhanced + self.position1(enhanced)
        normalized = self.norm1(positioned.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        gate = F.silu(self.gate(normalized))
        projected = F.silu(self.input_depthwise(self.input_projection(normalized)))
        qx, kx, value, qy, ky = self.qkv(projected).chunk(5, dim=1)
        axial = self.horizontal(qx, kx, value)
        axial = axial + self.vertical(qy, ky, value) + self.local(value)
        output = positioned + self.output_projection(axial * gate)
        output = output + self.position2(output)
        normalized2 = self.norm2(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return output + self.mlp(normalized2)

