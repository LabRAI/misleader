from cifar10_models import *
import network
import random

def get_classifier(args, classifier, pretrained=True, num_classes=10):
    if classifier.lower()=='lenet5':
        return network.lenet.LeNet5()
    elif classifier.lower()=='lenet5half':
        return network.lenet.LeNet5Half()
    elif classifier.lower()=='lenet5fifth':
        return network.lenet.LeNet5Fifth()
    elif classifier == 'vgg11_bn':
        return vgg11_bn(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg13_bn':
        return vgg13_bn(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg16_bn':
        print('using VGG16_BN network as teacher network')
        return vgg16_bn(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg19_bn':
        return vgg19_bn(pretrained=pretrained, num_classes=num_classes)
    if classifier == 'vgg11':
        return vgg11(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg13':
        return vgg13(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg16':
        print('using VGG16 as teacher network')
        return vgg16(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg19':
        return vgg19(pretrained=pretrained, num_classes=num_classes)
    # elif classifier == 'resnet18':
    #     return resnet18(pretrained=pretrained, num_classes=num_classes)
    # elif classifier == 'resnet34':
    #     return resnet34(pretrained=pretrained, num_classes=num_classes)
    # elif classifier == 'resnet50':
    #     return resnet50(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'densenet121':
        print('Using densenet121 as the teacher network')
        return densenet121(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'densenet161':
        return densenet161(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'densenet169':
        return densenet169(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'mobilenet_v2':
        print('using mobilenet_v2')
        return mobilenet_v2(pretrained=pretrained, num_classes=num_classes)
    elif classifier == "resnet50_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return network.resnet_8x.ResNet50_8x(num_classes=num_classes)
    elif classifier == "resnet34_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    elif classifier == "resnet18_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet18_8x from here")
        print('loading resnet18_8X')
        return network.resnet_8x.ResNet18_8x(num_classes=num_classes)

    else:
        raise NameError('Please enter a valid classifier')


classifiers = [
    'lenet5',
    'lenet5half',
    'lenet5fifth',
    "resnet34_8x", # Default DFAD
    "resnet50_8x", # Default DFAD
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet161",
    "densenet169",
    "mobilenet_v2",
    "googlenet",
    "inception_v3",
    "wrn-28-10",
    "resnet18_8x",
    "kt-wrn-40-2",
    "kt-wrn-28-2-new",
]
