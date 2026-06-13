

import torch

from torchvision import transforms

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# PREPARE INPUT TENSOR
# =====================================================

def prepare_input(image):

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict(image, model, idx2label):

    image_tensor = prepare_input(image)

    device = next(
        model.parameters()
    ).device

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    confidence, pred_idx = torch.max(
        probabilities,
        dim=1
    )

    prediction = idx2label[
        str(pred_idx.item())
    ]

    return (
        prediction,
        confidence.item(),
        probabilities.squeeze().cpu().numpy()
    )

# =====================================================
# TOP-K PREDICTIONS
# =====================================================

def get_top_predictions(
    probabilities,
    idx2label,
    top_k=3
):

    probs_tensor = torch.tensor(
        probabilities
    )

    values, indices = torch.topk(
        probs_tensor,
        k=top_k
    )

    results = []

    for value, idx in zip(
        values,
        indices
    ):

        results.append({

            "disease":
            idx2label[
                str(idx.item())
            ],

            "probability":
            float(
                value.item() * 100
            )

        })

    return results

# =====================================================
# FULL PROBABILITY TABLE
# =====================================================

def get_probability_dict(
    probabilities,
    idx2label
):

    result = {}

    for i, prob in enumerate(
        probabilities
    ):

        result[
            idx2label[str(i)]
        ] = round(
            float(prob * 100),
            2
        )

    return result


