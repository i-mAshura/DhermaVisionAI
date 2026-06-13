import torch
import timm

MODEL_NAME = "swin_tiny_patch4_window7_224"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

def load_model(model_path):

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=8
    )

    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    return model