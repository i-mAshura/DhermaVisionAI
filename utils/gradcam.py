import numpy as np

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

def reshape_transform(
    tensor,
    height=7,
    width=7
):

    result = tensor.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2)
    )

    result = result.transpose(
        2,
        3
    ).transpose(
        1,
        2
    )

    return result

def generate_gradcam(
    model,
    image,
    input_tensor
):

    target_layers = [
        model.layers[-1].blocks[-1].norm2
    ]

    rgb_img = np.array(
        image.resize((224,224))
    )

    rgb_img = rgb_img.astype(
        np.float32
    ) / 255.0

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization