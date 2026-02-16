import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTemporalGraphical(nn.Module):
    """
    The core ST-GCN operation:
    - temporal conv + graph conv using K adjacency partitions
    Input: (N, C, T, V)
    A: (K, V, V)
    """
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=9, t_stride=1, t_padding=4, bias=True):
        super().__init__()
        assert kernel_size > 0
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            bias=bias
        )

    def forward(self, x, A):
        x = self.conv(x)  # (N, outC*K, T, V)
        N, KC, T, V = x.size()
        K = A.size(0)
        x = x.view(N, K, KC // K, T, V)  # (N, K, outC, T, V)
        x = torch.einsum("nkctv,kvw->nctw", x, A)  # sum over K partitions
        return x.contiguous()

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.2, kernel_size=3):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        K = A.shape[0]

        self.gcn = ConvTemporalGraphical(
            in_channels, out_channels,
            kernel_size=K,
            t_kernel_size=9, t_stride=1, t_padding=4
        )

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), padding=(4,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x, self.A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_class=35, in_channels=2, A=None, num_joints=21):
        super().__init__()
        assert A is not None, "Pass adjacency A from Graph().A"
        self.num_joints = num_joints

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.l1 = STGCNBlock(in_channels, 64, A, stride=1, residual=False)
        self.l2 = STGCNBlock(64, 64, A, stride=1)
        self.l3 = STGCNBlock(64, 64, A, stride=1)
        self.l4 = STGCNBlock(64, 128, A, stride=2)
        self.l5 = STGCNBlock(128, 128, A, stride=1)
        self.l6 = STGCNBlock(128, 256, A, stride=2)
        self.l7 = STGCNBlock(256, 256, A, stride=1)

        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        x = x.view(N, V * C, T)                 # (N, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # Global average pool over (T,V)
        x = x.mean(dim=2).mean(dim=2)  # (N, 256)
        return self.fc(x)
