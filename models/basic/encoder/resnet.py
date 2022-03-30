import torchvision.models as models


class ResNet(models.resnet.ResNet):
    """resnet model without avgpool and fc"""
 
    def __init__(self, block, layers, **kwargs):
        super(ResNet, self).__init__(block, layers, **kwargs)
        del self.avgpool, self.fc
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # stage 1, stride 4, by default -----
        s1 = self.layer1(x)
        # stage 2, stride 8, by default -----
        s2 = self.layer2(s1)
        # stage 3, stride 16 by default -----
        s3 = self.layer3(s2)
        # stage 4, stride 32 by default -----
        s4 = self.layer4(s3)
        # return all stages -----------------
        return s1, s2, s3, s4
    

def resnet50():
    "resnet50 without avgpool and fc, output stride 32"
    return ResNet(models.resnet.Bottleneck, [3, 4, 6, 3])

def seg_resnet50():
    "resnet50 without avgpool and fc, layer4 stride replaced with dilation, output stride 16"
    return ResNet(models.resnet.Bottleneck, [3, 4, 6, 3], 
                  replace_stride_with_dilation=[False, False, True])

def resnet101():
    "resnet101 without avgpool and fc, output stride 32"
    return ResNet(models.resnet.Bottleneck, [3, 4, 23, 3])