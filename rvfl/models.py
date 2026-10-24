"""Model architectures; get_model(name) builds one by name. The paper uses ResNet20 on CIFAR-10."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union, List, Optional, Dict, cast
import math

# parameter size = 2,193,226
class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32x32 -> 32x32
        self.pool = nn.MaxPool2d(2, 2)                            # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)# 16x16 -> 16x16
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 32, 32, 32]
        x = F.relu(self.conv2(x))  # [B, 64, 32, 32]
        x = self.pool(x)           # [B, 64, 16, 16]
        x = F.relu(self.conv3(x))  # [B, 128, 16, 16]
        x = self.pool(x)           # [B, 128, 8, 8]
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
# parameter size = 259,960
class MNIST_FLTrust(nn.Module):
    def __init__(self):
        super(MNIST_FLTrust, self).__init__()
        # Input: 1x28x28, Output: 30x14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 30x14x14, Output: 50x7x7
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 50 * 7 * 7 = 2450
        self.fc1 = nn.Linear(50 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, 50 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels  = in_planes,
        out_channels = out_planes,
        kernel_size  = 3,
        stride       = stride,
        padding      = dilation,
        groups       = groups,
        bias         = False,
        dilation     = dilation
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels  = in_planes,
        out_channels = out_planes,
        kernel_size  = 1,
        stride       = stride,
        bias         = False
    )

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.GroupNorm
        
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1      = conv1x1(inplanes, width)
        self.bn1        = norm_layer(width//16, width)
        self.conv2      = conv3x3(width, width, stride, groups, dilation)
        self.bn2        = norm_layer(width//16, width)
        self.conv3      = conv1x1(width, planes * self.expansion)
        self.bn3        = norm_layer((planes * self.expansion)//16, planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes//16, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes//16, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
        cifar_mode: bool = False
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer
        self.cifar_mode = cifar_mode

        self.inplanes = 16 if cifar_mode else 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        self.groups = groups
        self.base_width = width_per_group
        
        if self.cifar_mode:
            self.conv1 = nn.Conv2d(
                in_channels  = 3, 
                out_channels = self.inplanes, 
                kernel_size  = 3, 
                stride       = 1, 
                padding      = 1,
                bias         = False
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels  = 3, 
                out_channels = self.inplanes, 
                kernel_size  = 7, 
                stride       = 2, 
                padding      = 3,
                bias         = False
            )
            
        self.bn1    = norm_layer(self.inplanes//16, self.inplanes)
        self.relu   = nn.ReLU(inplace=True)
        
        if not self.cifar_mode:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if self.cifar_mode:
            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilute=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilute=replace_stride_with_dilation[1])
            self.layer4 = None
            self.fc     = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.layer1 = self._make_layer(block, 64,  layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilute=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilute=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilute=replace_stride_with_dilation[2])
            self.fc     = nn.Linear(512 * block.expansion, num_classes)
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(
        self, 
        block: Type[nn.Module], 
        planes: int, 
        blocks: int, 
        stride: int = 1, 
        dilute: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilute:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer((planes * block.expansion)//16, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups     = self.groups,
                    base_width = self.base_width,
                    dilation   = self.dilation,
                    norm_layer = norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if not self.cifar_mode:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if not self.cifar_mode:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# parameter size = 272,474
def resnet20(num_classes: int = 10, **kwargs) -> ResNet:
    """Constructs a ResNet-20 model for CIFAR-10."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, cifar_mode=True, **kwargs)


# parameter size = 11,181,642
def resnet18(num_classes: int = 10, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34(num_classes: int = 10, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet50(num_classes: int = 10, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet101(num_classes: int = 10, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def resnet152(num_classes: int = 10, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model."""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.GroupNorm(v//16, v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        pool_size: int = 7,
        hidden_dim: int = 4096
    ) -> None:
        super(VGG, self).__init__()
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * pool_size * pool_size, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg11_bn(num_classes: int = 1000, **kwargs) -> VGG:
    return VGG(make_layers(cfgs['vgg11'], batch_norm=True), num_classes=num_classes, **kwargs)

def vgg13_bn(num_classes: int = 1000, **kwargs) -> VGG:
    return VGG(make_layers(cfgs['vgg13'], batch_norm=True), num_classes=num_classes, **kwargs)

def vgg16_bn(num_classes: int = 1000, **kwargs) -> VGG:
    return VGG(make_layers(cfgs['vgg16'], batch_norm=True), num_classes=num_classes, **kwargs)

def vgg19_bn(num_classes: int = 1000, **kwargs) -> VGG:
    return VGG(make_layers(cfgs['vgg19'], batch_norm=True), num_classes=num_classes, **kwargs)


def get_model(model_name: str, num_classes: int = 10, input_size: int = 32):
    if model_name == 'MNIST_FLTrust':
        return MNIST_FLTrust()
    elif model_name == 'CIFAR_CNN':
        return CIFAR_CNN()
    elif 'ResNet' in model_name:
        if model_name == 'ResNet18':
            return resnet18(num_classes)
        elif model_name == 'ResNet20':
            return resnet20(num_classes)
        elif model_name == 'ResNet34':
            return resnet34(num_classes)
        elif model_name == 'ResNet50' or model_name == 'ResNet':
            return resnet50(num_classes)
        elif model_name == 'ResNet101':
            return resnet101(num_classes)
        elif model_name == 'ResNet152':
            return resnet152(num_classes)
    elif 'VGG' in model_name:
        pool_size = max(1, input_size//32)
        d_in = 512 * pool_size * pool_size
        geometric_mean = math.sqrt(d_in * num_classes)
        hidden_dim = min(4096, max(256, 2 ** math.ceil(math.log2(geometric_mean))))
        if model_name == 'VGG11':
            return vgg11_bn(num_classes, pool_size=pool_size, hidden_dim=hidden_dim)
        elif model_name == 'VGG13':
            return vgg13_bn(num_classes, pool_size=pool_size, hidden_dim=hidden_dim)
        elif model_name == 'VGG16':
            return vgg16_bn(num_classes, pool_size=pool_size, hidden_dim=hidden_dim)
        elif model_name == 'VGG19':
            return vgg19_bn(num_classes, pool_size=pool_size, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown model name: {model_name}")
    