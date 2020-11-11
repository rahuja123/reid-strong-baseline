from torch import nn
from torch.autograd import Function
from .backbones.resnetattn import resnet50_CBAM_twin_tail_v5

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class ResNet50_CBAM_TwinTail_v5(nn.Module):
    def __init__(self, num_classes_1, num_classes_2, last_stride, pooling, bnneck, reverse_grad=False, IN=False, in_planes=2048, model_name='ResNet50TwoTask'):
        super(ResNet50_CBAM_TwinTail_v5, self).__init__()
        self.base = resnet50_CBAM_twin_tail_v5(pretrained=True, last_stride=last_stride, IN=IN)
        self.in_planes = in_planes
        self.model_name = model_name

        if pooling == 'AVG':
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'MAX':
            self.gap = nn.AdaptiveMaxPool2d(1)
        else:
            raise Exception('The POOL value should be AVG or MAX')
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2

        self.bnneck = bnneck

        if self.bnneck:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

        self.id_classifier = nn.Linear(self.in_planes, self.num_classes_1, bias=False)
        self.id_classifier.apply(weights_init_classifier)
        # 256 because we cashed-out the cam-id feature after conv_block1 of the resnet50
        self.cam_classifier = nn.Linear(256, self.num_classes_2, bias=False)
        self.cam_classifier.apply(weights_init_classifier)

        self.reverse_grad = reverse_grad

        if self.reverse_grad:
            print('Using reverse grad layer...')
            self.revgrad = GradReverse.apply

    def forward(self, x):
        id_feat, cam_feat = self.base(x) # (b, 2048, 1, 1) for id_feat, (b, 256, 1, 1) for cam_feat
        global_id_feat = self.gap(id_feat)  
        global_id_feat = global_id_feat.view(global_id_feat.shape[0], -1)  # flatten to (bs, 2048)
        global_cam_feat = self.gap(cam_feat)
        cam_feat = global_cam_feat.view(global_cam_feat.shape[0], -1)  # flatten to (bs, 256)
        
        id_feat = self.bottleneck(global_id_feat) if self.bnneck else global_id_feat
        # cam_feat = self.bottleneck(global_cam_feat) if self.bnneck else global_cam_feat
        if self.training:
            id_score = self.id_classifier(id_feat)
            if self.reverse_grad:
                cam_feat = self.revgrad(cam_feat)
            cam_score = self.cam_classifier(cam_feat)
            return id_score, cam_score, global_id_feat  # global feature for triplet loss
        else:
            return id_feat
