"""Student family (own design), projectors and the frozen teacher."""
import torch
import torch.nn as nn

from . import config as C


def _div8(x):
    return max(8, int(round(x / 8)) * 8)


class BasicBlock(nn.Module):
    """Two 3x3 convs + BN + ReLU with an identity (or 1x1-conv) shortcut."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class StudentNet(nn.Module):
    """Small residual convnet for 64-96 px inputs.

    - stem: one 3x3 conv, stride 2, no max-pool (a ResNet 7x7/2 + max-pool stem
      would divide a 64 px image by 4 before any residual block);
    - 4 stages of ``DEPTHS`` basic blocks; stages 2-4 downsample by 2
      (64 px input -> 32, 16, 8, 4 feature maps);
    - BatchNorm + ReLU everywhere; global average pooling; no head.
    The size family scales ONLY the channel widths by ``width_mult``.
    """

    def __init__(self, width_mult, base=C.BASE_WIDTHS, depths=C.DEPTHS):
        super().__init__()
        w = [_div8(b * width_mult) for b in base]
        self.widths = w
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        layers, cin = [], w[0]
        for i, (cout, d) in enumerate(zip(w, depths)):
            for j in range(d):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BasicBlock(cin, cout, stride))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = w[-1]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.pool(self.blocks(self.stem(x))).flatten(1)


def build_student(size):
    return StudentNet(C.SIZES[size]).to(memory_format=torch.channels_last)


def vicreg_projector(in_dim, hidden=C.PROJ_HIDDEN, out=C.VICREG_OUT):
    """3-layer expander as in VICReg (scaled down for a small student)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, out))


def distill_projector(in_dim, teacher_dim, hidden=C.PROJ_HIDDEN):
    """Projector from the project statement (Section 3.5). Discarded after training."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                         nn.Linear(hidden, teacher_dim))


def load_teacher(pretrained=None):
    """Frozen self-supervised ViT. Returns (model, mean, std, feat_dim).
    With num_classes=0 timm returns the final-norm [CLS] token (DINO's feature)."""
    import timm
    pretrained = C.TEACHER_PRETRAINED if pretrained is None else pretrained
    model = timm.create_model(C.TEACHER_NAME, pretrained=pretrained, num_classes=0)
    cfg = timm.data.resolve_data_config({}, model=model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tuple(cfg["mean"]), tuple(cfg["std"]), model.num_features


def n_params(model):
    return sum(p.numel() for p in model.parameters())
