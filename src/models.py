"""
models.py
---------
Model definitions for the retina age prediction project.

One configurable CNN family (RetinaCNN) is used for every deep
configuration. Keeping a single family matters for the comparisons: two
configs that differ only in the argument we are studying really do
differ only in that argument.

The head deserves an explanation, because it is what makes the padding
comparison fair. Our first version flattened the last feature map
straight into a linear layer. That means the spatial size of the feature
map decides the size of the first FC layer - and "valid" padding shrinks
the spatial size at every conv, so the valid model ended up with 26%
fewer parameters than the same model. We were then comparing capacity,
not padding. Putting an AdaptiveAvgPool2d((POOL, POOL)) before the head
pins the flattened size to channels x POOL x POOL regardless of what the
convolutions did to the spatial dimensions, so both padding modes give
byte-identical parameter counts.

The same problem in a different form affects the filter-size comparison:
a 7x7 filter has roughly 5.4x the weights of a 3x3 filter, so a naive
7x7 network is simply a bigger model. There we use the `width` argument
to scale the channel counts down until the totals match (see
match_width_for_params below).
"""

import torch
import torch.nn as nn

BASE_CHANNELS = [16, 32, 64, 128]
POOL = 4  # adaptive pool output side; head size is channels * POOL * POOL


class RetinaCNN(nn.Module):
    """
    Conv blocks (conv -> optional BN -> ReLU -> optional Dropout2d -> maxpool),
    then adaptive pooling and a small FC head.

    filter_size       3 or 7 - the receptive field experiment
    num_blocks        how many conv blocks
    padding           "same" keeps the border, "valid" discards it
    dropout_position  "none" / "early" (Dropout2d between conv blocks)
                      / "late" (Dropout in the FC head, the usual place)
    width             multiplier on the channel counts, used to match
                      parameter counts across configurations
    use_batchnorm     BatchNorm was not covered in the course; it is on by
                      default because it stabilises training, and the flag
                      exists so we can show what it costs and report that
                      honestly.
    """

    def __init__(self, filter_size=3, num_blocks=4, padding="same",
                 dropout_position="late", dropout_rate=0.3, width=1.0,
                 img_size=224, use_batchnorm=True):
        super().__init__()
        assert padding in ("same", "valid")
        assert dropout_position in ("none", "early", "late")
        assert 1 <= num_blocks <= len(BASE_CHANNELS)

        pad = filter_size // 2 if padding == "same" else 0
        chans = [max(4, int(round(c * width))) for c in BASE_CHANNELS[:num_blocks]]
        chans = [3] + chans

        layers = []
        for i in range(num_blocks):
            layers.append(nn.Conv2d(chans[i], chans[i + 1], kernel_size=filter_size, padding=pad))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(chans[i + 1]))
            layers.append(nn.ReLU())
            # Dropout2d, not plain Dropout: neighbouring pixels in a feature
            # map are strongly correlated, so dropping individual activations
            # barely removes information. Dropout2d drops whole channels.
            if dropout_position == "early" and i < num_blocks - 1:
                layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*layers)

        # This is the piece that makes "same" and "valid" comparable.
        self.pool = nn.AdaptiveAvgPool2d((POOL, POOL))

        self.fc1 = nn.Linear(chans[-1] * POOL * POOL, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_position == "late" else nn.Identity()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def match_width_for_params(target_params, tolerance=0.05, **config):
    """
    Finds the `width` that brings a configuration within `tolerance` of
    target_params, by bisection on the multiplier.

    This is how the 7x7 network is made comparable to the 3x3 one: a 7x7
    filter carries 49/9 times the weights of a 3x3 filter, so we shrink
    the channel counts until the two networks are the same size and the
    only thing left differing is the receptive field.

    Returns (width, params, relative_error).
    """
    lo, hi = 0.05, 4.0
    best = None
    for _ in range(60):
        mid = (lo + hi) / 2
        n = count_params(RetinaCNN(width=mid, **config))
        err = (n - target_params) / target_params
        if best is None or abs(err) < abs(best[2]):
            best = (mid, n, err)
        if n > target_params:
            hi = mid
        else:
            lo = mid
    if abs(best[2]) > tolerance:
        raise ValueError(
            f"could not match {target_params:,} params within {tolerance:.0%} "
            f"(best {best[1]:,}, off by {best[2]:+.1%})"
        )
    return best


if __name__ == "__main__":
    # Sanity check: confirm the two fairness mechanisms actually work.
    control = RetinaCNN()
    n_control = count_params(control)
    print(f"control (3x3, 4 blocks, same, late) : {n_control:,} params")

    valid = RetinaCNN(padding="valid")
    print(f"valid padding                       : {count_params(valid):,} params "
          f"-> identical: {count_params(valid) == n_control}")

    early = RetinaCNN(dropout_position="early")
    print(f"early dropout                       : {count_params(early):,} params "
          f"-> identical: {count_params(early) == n_control}")

    w, n, err = match_width_for_params(n_control, filter_size=7, num_blocks=3)
    print(f"7x7 shallow, width={w:.4f}            : {n:,} params ({err:+.2%} vs control)")

    x = torch.randn(2, 3, 224, 224)
    for name, m in [("control", control), ("valid", valid), ("early", early),
                    ("7x7 matched", RetinaCNN(filter_size=7, num_blocks=3, width=w))]:
        print(f"  {name:12s} output {tuple(m(x).shape)}")
