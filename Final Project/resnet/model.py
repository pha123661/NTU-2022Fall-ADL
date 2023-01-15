import torch
import torch.nn as nn
import torchvision.models as models
import parser
import clip
from PIL import Image


def get_model():
    args = parser.arg_parse()
    if args.model == "resnet34":
        print("===> loading resnet34...")

        model = models.resnet34(pretrained=True)

    elif args.model == "resnet50":
        print("===> loading resnet50...")

        model = models.resnet50(pretrained=True)

    elif args.model == "resnet101":
        print("===> loading resnet101...")

        model = models.resnet50(pretrained=True)

    else:
        print("===> loading clip resnet...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model, preprocess = clip.load("ViT-B/32", device=device)
        model, preprocess = clip.load("RN50", device=device)
        model = model.visual

    print("===> changing last layer...")

    # change last layer to 15
    num_features = model.fc.in_features if args.model != "clip" else 512
    model.fc = nn.Sequential(
        nn.Dropout(0.8),
        nn.Linear(num_features, 15)
    )

    # print(model)
    '''
    # num_features = model.classifier[6].in_features
    # features = list(model.classifier[6].children())[:-1] # Remove last layer
    # features.extend([nn.Linear(num_features, 50)]) # Add our layer with 50 outputs
    # model.classifier = nn.Sequential(*features) # Replace the model classifier
    # print(model)
    '''
    return model
