import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mamba_ssm import Mamba
from torch.nn import functional as F

class DiffNormalized(nn.Module):
    """
    Added DiffNormalized preprocessing as mentioned in the paper (Section 4.2).
    This enhances the model's robustness to motion and background pixels.
    """
    def __init__(self):
        super(DiffNormalized, self).__init__()
        
    def forward(self, x):
        # Calculate inter-frame differences as (Xt+1 - Xt)/(Xt + Xt+1)
        x_diff = (x[:, :, 1:] - x[:, :, :-1]) / (x[:, :, 1:] + x[:, :, :-1] + 1e-7)
        
        # Normalize by standard deviation
        std = torch.std(x_diff, dim=(2, 3, 4), keepdim=True)
        x_normalized = x_diff / (std + 1e-7)
        
        # Pad to maintain temporal dimension
        padding = torch.zeros_like(x[:, :, 0:1])
        return torch.cat([padding, x_normalized], dim=2)

class ChannelAttention3D(nn.Module):
    """
    Channel Attention module to enhance channel representation and reduce redundancy.
    Maintained original implementation but improved integration in the block structure.
    """
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x*attention

class LateralConnection(nn.Module):
    """
    Enhanced implementation of LateralConnection for better feature fusion
    between fast and slow streams as specified in the paper (Section 3.2).
    """
    def __init__(self, fast_channels=32, slow_channels=64):
        super(LateralConnection, self).__init__()
        # Using temporal convolution with kernel size 3x1x1 as specified in the paper
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1,0,0]),   
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )
        
    def forward(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path

class CDC_T(nn.Module):
    """
    Maintained original CDC_T class name but improved implementation
    to align with the Temporal Difference Convolution concept in the paper.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.5):
        # Updated default theta to 0.5 as per paper's optimal setting (Section 4.5)
        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # Regular 3D convolution output
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # Only apply temporal difference when temporal kernel size > 1
            if t > 1:
                # Enhanced temporal difference calculation
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                # Implemented as per Equation (5) in the paper
                return out_normal - self.theta * out_diff
            else:
                return out_normal

class BiMambaLayer(nn.Module):
    """
    Implementation of Bidirectional Mamba layer for capturing
    long-range dependencies in both temporal directions.
    This is based on the paper's description of Temporal Bidirectional Mamba (Section 3.3).
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super(BiMambaLayer, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.drop_path = nn.Identity()
        
        # Forward direction Mamba
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Backward direction Mamba
        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Output layer norm
        self.norm_out = nn.LayerNorm(dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # Input shape: (B, L, D)
        
        # Input normalization
        x_norm = self.norm(x)
        
        # Forward direction
        y_forward = self.forward_mamba(x_norm)
        
        # Backward direction (flip sequence, process, then flip back)
        x_backward = torch.flip(x_norm, dims=[1])
        y_backward = self.backward_mamba(x_backward)
        y_backward = torch.flip(y_backward, dims=[1])
        
        # Combine outputs from both directions
        y_combined = y_forward + y_backward
        
        # Apply output normalization and residual connection
        x_out = self.norm_out(x + self.drop_path(y_combined))
        
        return x_out

class MambaLayer(nn.Module):
    """
    Maintained MambaLayer class with enhancements to support
    the Temporal Difference Mamba concept from the paper.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        # Using bimamba=True to enable bidirectional processing as per paper
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba=True,     # Enable bidirectional processing
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        B, C, nf, H, W = x.shape
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        out = x_out.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out 

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out

class TemporalDifferenceMambaBlock(nn.Module):
    """
    Implementation of the TD-Mamba block as described in the paper (Section 3.3).
    Combines CDC_T, BiMambaLayer, and ChannelAttention3D.
    """
    def __init__(self, channels, theta=0.5, d_state=16, d_conv=4, expand=2):
        super(TemporalDifferenceMambaBlock, self).__init__()
        
        # Temporal Difference Convolution
        self.tdc = CDC_T(channels, channels, theta=theta)
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Bidirectional Mamba
        self.bi_mamba = BiMambaLayer(
            dim=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Channel Attention
        self.channel_attention = ChannelAttention3D(channels, reduction=2)
        
    def forward(self, x):
        # Input: (B, C, T, H, W)
        
        # Temporal Difference Convolution
        x = self.tdc(x)
        x = self.relu(self.bn(x))
        
        # Reshape for Mamba processing
        B, C, T, H, W = x.shape
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, T*H*W, C)
        
        # Apply bidirectional Mamba
        x_mamba = self.bi_mamba(x_flat)
        
        # Reshape back to 5D tensor
        x_mamba = x_mamba.permute(0, 2, 1).reshape(B, C, T, H, W)
        
        # Apply channel attention
        x_out = self.channel_attention(x_mamba)
        
        return x_out

def conv_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, activation='relu'):
    """Helper function to create a conv block. Unchanged."""
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

class PhysMamba(nn.Module):
    """
    Enhanced PhysMamba implementation maintaining the original structure
    but incorporating the improvements from the paper.
    """
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super(PhysMamba, self).__init__()
        
        # Added DiffNormalized preprocessing
        self.diff_normalized = DiffNormalized()

        # Kept original structure but organized into a stem block
        self.ConvBlock1 = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])  
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Replaced original Block1-6 with enhanced TemporalDifferenceMambaBlocks
        # Slow Stream
        self.Block1 = TemporalDifferenceMambaBlock(64, theta)
        self.Block2 = TemporalDifferenceMambaBlock(64, theta)
        self.Block3 = TemporalDifferenceMambaBlock(64, theta)
        # Fast Stream
        self.Block4 = TemporalDifferenceMambaBlock(32, theta)
        self.Block5 = TemporalDifferenceMambaBlock(32, theta)
        self.Block6 = TemporalDifferenceMambaBlock(32, theta)

        # Upsampling blocks - maintained as in original
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1), mode='nearest'),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1), mode='nearest'),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        # Final layers - maintained as in original
        self.ConvBlockLast = nn.Conv3d(48, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # Enhanced lateral connections
        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        # Dropout layers - maintained as in original
        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x): 
        [batch, channel, length, width, height] = x.shape

        # Apply DiffNormalized preprocessing
        x = self.diff_normalized(x)

        # Shallow stem processing (same as original)
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x) 
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)  
        x = self.MaxpoolSpa(x) 
    
        # Process streams
        s_x = self.ConvBlock4(x) # Slow stream 
        f_x = self.ConvBlock5(x) # Fast stream 

        # First set of blocks and fusion
        s_x1 = self.Block1(s_x)
        s_x1 = self.MaxpoolSpa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.Block4(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1, f_x1) # Enhanced lateral connection

        # Second set of blocks and fusion
        s_x2 = self.Block2(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)
        s_x2 = self.drop_3(s_x2)
        
        f_x2 = self.Block5(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2, f_x2) # Enhanced lateral connection
        
        # Third blocks and upsampling
        s_x3 = self.Block3(s_x2) 
        s_x3 = self.upsample1(s_x3) 
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.Block6(f_x2)
        f_x3 = self.ConvBlock6(f_x3) 
        f_x3 = self.drop_6(f_x3)

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1) 
        x_final = self.upsample2(x_fusion) 

        x_final = self.poolspa(x_final)
        x_final = self.ConvBlockLast(x_final)

        rPPG = x_final.view(-1, length)

        return rPPG

class NegPearsonLoss(nn.Module):
    """
    Implementation of negative Pearson loss as described in Section 3.4.
    This ensures alignment between predicted and ground truth rPPG signals.
    """
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, preds, labels):
        """
        Calculate negative Pearson correlation as per Equation (7) in the paper
        
        Args:
            preds: predicted rPPG signals, shape [batch_size, sequence_length]
            labels: ground truth rPPG signals, shape [batch_size, sequence_length]
            
        Returns:
            loss: negative Pearson correlation (higher correlation -> lower loss)
        """
        T = preds.size(1)
        loss = 0
        batch_size = preds.size(0)
        
        for i in range(batch_size):
            pred = preds[i]
            label = labels[i]
            
            pred_mean = torch.mean(pred)
            label_mean = torch.mean(label)
            
            # Numerator
            numerator = T * torch.sum(pred * label) - torch.sum(pred) * torch.sum(label)
            
            # Denominator
            term1 = torch.sqrt(T * torch.sum(pred**2) - torch.sum(pred)**2)
            term2 = torch.sqrt(T * torch.sum(label**2) - torch.sum(label)**2)
            denominator = term1 * term2
            
            # Pearson correlation
            pearson = numerator / (denominator + 1e-7)
            
            # Negative Pearson for minimization
            loss += 1 - pearson
            
        return loss / batch_size