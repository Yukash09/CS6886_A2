import torch.nn as nn 
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import cast

def get_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # We will change the stride = 1 for the first layer since CIFAR-10, which has 32x32 images. This will retain resolution in the early layers. 
    fblock = cast(nn.Conv2d , cast(nn.Sequential, model.features[0])[0])

    cast(nn.Sequential , model.features[0])[0] = nn.Conv2d(
        in_channels=fblock.in_channels,
        out_channels=fblock.out_channels,
        kernel_size=cast(tuple[int, int], fblock.kernel_size),
        stride=1,
        padding=cast(tuple[int, int], fblock.padding),
        bias=(fblock.bias is not None)
    )

    cast(nn.Sequential , model.features[0])[0].weight.data = fblock.weight.data.clone()
    if fblock.bias is not None:
        cast(nn.Sequential, model.features[0])[0].bias.data = fblock.bias.data.clone()

    # We need to change the last layer to output one of the 10 classes:
    # Airplane Automobile Bird Cat Deer Dog Frog Horse Ship Truck
    dropout = cast(nn.Dropout, model.classifier[0]).p
    in_features = cast(nn.Linear, model.classifier[1]).in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features=in_features, out_features=10)
    )

    return model