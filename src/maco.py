import random
from typing import Literal

import torch
import torchvision
from tqdm import tqdm


class NormalRandomResizedCrop:
    """Randomly crops and optionally resizes an image using a normal distribution for the crop ratio."""

    def __init__(
        self,
        mean: float = 0.25,
        std: float = 0.1,
        resize_to: tuple[int, int] = (224, 224),
    ):
        """Initializes the NormalRandomResizedCrop transformation.

        Args:
            mean (float, optional): The mean ratio for the crop size relative to the image dimensions.
            std (float, optional): The standard deviation of the crop size ratio.
            resize_to (tuple): The target size to resize the cropped image (e.g., (224, 224)).
                If None, no resizing is performed.

        """
        assert len(resize_to) == 2
        self.mean = mean
        self.std = std
        self.resize_to = resize_to

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Randomly crops and optionally resizes the input image using a normal distribution for the crop ratio.

        The crop ratio is sampled from a normal distribution N(mean, std) and clamped between 0.1 and 1.0.
        The image is cropped at a random position and then resized if a target size is provided.

        Args:
            image (torch.Tensor): shape (C, H, W) or (N, C, H, W)
                The input image tensor.

        Returns:
            torch.Tensor: shape (C, H, W) or (N, C, H, W)
                The cropped (and possibly resized) image tensor. The output shape will match the input
                dimensions (3D or 4D) accordingly.

        """
        assert image.ndim == 3 or image.ndim == 4

        H, W = torchvision.transforms.functional.get_image_size(image)  # noqa: N806

        # sample the crop ratio from a normal distribution N(mean, std) and clamp between 0.1 and 1.0.
        crop_frac = random.gauss(self.mean, self.std)
        crop_frac = max(0.1, min(1.0, crop_frac))

        # calculate the crop height and width based on the image dimensions.
        crop_h = int(H * crop_frac)
        crop_w = int(W * crop_frac)

        # choose a random crop position ensuring the crop is within image bounds.
        left = random.randint(0, W - crop_w) if W - crop_w > 0 else 0
        top = random.randint(0, H - crop_h) if H - crop_h > 0 else 0

        # determine the crop boundaries.
        bottom = top + crop_h
        right = left + crop_w

        # crop the image using tensor slicing.
        batched = image.unsqueeze(0) if image.ndim == 3 else image
        cropped = batched[:, :, top:bottom, left:right]

        # resize the cropped image using torch.nn.functional.interpolate.
        resized = torch.nn.functional.interpolate(
            cropped, size=self.resize_to, mode="bilinear", align_corners=False
        )

        return resized.squeeze(0) if image.ndim == 3 else resized


def recorrelate_colors(images: torch.Tensor) -> torch.Tensor:
    """Map decorrelated colors to normal colors using the empirical ImageNet color correlation matrix.

    This function borrows ideas from original implementation:
    https://github.com/deel-ai/xplique/blob/master/xplique/features_visualizations/preconditioning.py#L20-L46

    Args:
        images (torch.Tensor): shape (3, H, W)
            An input image tensor.

    Returns:
        torch.Tensor: shape (3, H, W)
            An image tensor with re-correlated colors.

    """
    assert images.ndim == 3
    assert images.size(0) == 3

    # empirical color correlation matrix for ImageNet.
    imagenet_color_correlation = torch.tensor(
        [
            [0.56282854, 0.58447580, 0.58447580],
            [0.19482528, 0.00000000, -0.19482528],
            [0.04329450, -0.10823626, 0.06494176],
        ],
        dtype=images.dtype,
        device=images.device,
    )

    # convert from (3, H, W) to (H, W, 3).
    images_permuted = images.permute(1, 2, 0)

    # flatten the image to shape (H*W, 3).
    images_flat = images_permuted.reshape(-1, 3)

    # apply linear transformation using the color correlation matrix.
    images_flat = torch.matmul(images_flat, imagenet_color_correlation)

    # eeshape back to the original shape (H, W, 3).
    images_permuted = images_flat.reshape(images_permuted.shape)

    # convert back to (3, H, W) and return.
    return images_permuted.permute(2, 0, 1)


def compute_unnormalized_average_magnitude(
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Compute the unnormalized average magnitude of the Fourier transform across an entire dataset.

    This function iterates over the provided dataloader, applies a 2D Fast Fourier Transform (FFT)
    to each batch of images (on the specified device), and calculates the absolute value (magnitude)
    of the FFT coefficients. It then accumulates the sum of the magnitudes over all images and divides
    by the total number of samples to compute the average magnitude. The resulting tensor is moved to the CPU.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader that yields batches of images and labels.
            The images are expected to have values in the range [0, 1].
        device (torch.device): The device on which to perform the FFT computation.

    Raises:
        ValueError: If the dataset contained in the dataloader is empty.

    Returns:
        torch.Tensor: A tensor representing the average magnitude of the FFT over the dataset.
        The shape of the returned tensor corresponds to the image dimensions (e.g., [3, 224, 224]).

    """
    if len(dataloader.dataset) == 0:
        raise ValueError("The dataset is empty.")

    first_batch, _ = next(iter(dataloader))
    H, W = torchvision.transforms.functional.get_image_size(first_batch)  # noqa: N806

    total_magnitude = torch.zeros(3, H, W, device=device)
    total_samples = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            assert torch.all((images >= 0) & (images <= 1)).item()
            images = images.to(device)

            # apply FFT to last two dimensions.
            fft_images = torch.fft.fft2(images, norm="backward")
            magnitudes = torch.abs(fft_images)

            total_magnitude += magnitudes.sum(axis=0)
            total_samples += images.size(0)

    return (total_magnitude / total_samples).cpu()


def fourier_to_image(
    unnormalized_average_magnitude: torch.Tensor,
    phase: torch.Tensor,
    norm: Literal["backward", "forward", "ortho"] = "backward",
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reconstructs an image from Fourier magnitude and phase information.

    This function combines the provided unnormalized average magnitude with a phase
    tensor to create a complex Fourier spectrum. It then applies the inverse Fast Fourier
    Transform (IFFT) to convert the spectrum back to an image. The resulting image is
    normalized, its colors are recorrelated to match natural image statistics, and
    finally passed through a sigmoid function to ensure pixel values are in [0,1].

    This function borrows ideas from original implementation:
    https://github.com/deel-ai/xplique/blob/master/xplique/features_visualizations/preconditioning.py#L297-L331

    Args:
        unnormalized_average_magnitude (torch.Tensor): shape (3, H, W)
            A tensor containing the unnormalized average magnitude of Fourier coefficients.
        phase (torch.Tensor): shape (3, H, W)
            A tensor containing the phase information for Fourier coefficients.
        norm (Literal["backward", "forward", "ortho"], optional): Normalization mode for the IFFT.
            Should match the norm used when computing the average magnitude.
        eps (float, optional): Small epsilon value to prevent division by zero during normalization.

    Returns:
        torch.Tensor: shape (3, H, W)
            A tensor containing the reconstructed image in the range [0, 1].

    Raises:
        AssertionError: If the input tensors have incompatible shapes or incorrect
            number of dimensions.

    """
    assert phase.shape == unnormalized_average_magnitude.shape
    assert len(unnormalized_average_magnitude.shape) == 3
    assert unnormalized_average_magnitude.size(0) == 3

    phase = phase - phase.mean()
    phase = phase / (phase.std() + eps)

    spectrum = torch.polar(unnormalized_average_magnitude, phase)
    image = torch.real(torch.fft.ifft2(spectrum, norm=norm))

    image = image - image.mean()
    image = image / (image.std() + eps)

    image = recorrelate_colors(image)

    return torch.nn.functional.sigmoid(image)


def run_maco(
    model: torch.nn.Module,
    device: torch.device,
    unnormalized_average_magnitude: torch.Tensor,
    normalize_transform: torch.nn.Module,
    target_logit_idx: int = 0,
    num_steps: int = 256,
    learning_rate: float = 1.0,
    model_input_shape: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Run MACO algorithm.

    This function performs the MACO optimization algorithm to adjust the phase of a Fourier representation
    in order to maximize a specific target logit from the model's output. The process involves converting the
    Fourier representation to an image, applying normalization and random resized cropping, adding noise, and
    iteratively updating the phase using gradient descent.

    Args:
        model (torch.nn.Module): Neural network model used for generating logits.
        device (torch.device): Device to perform computations on.
        unnormalized_average_magnitude (torch.Tensor): shape (3, H, W)
            The unnormalized average magnitude tensor from the Fourier transform.
        normalize_transform (torch.nn.Module): A transformation module used to normalize the image.
        target_logit_idx (int, optional): Index of the target logit to maximize. Defaults to 0.
        num_steps (int, optional): Number of optimization steps. Defaults to 256.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1.0.
        model_input_shape (tuple[int, int]): The input shape of the model. Defaults to (224, 224).

    Returns:
        torch.Tensor: The optimized phase tensor.

    """
    assert len(model_input_shape) == 2

    model = model.to(device)
    unnormalized_average_magnitude = unnormalized_average_magnitude.to(device)

    noise_stds = torch.logspace(0, -4, steps=num_steps, dtype=torch.float32)

    phase = (
        2 * torch.pi * torch.rand(unnormalized_average_magnitude.shape, device=device)
    ) - torch.pi
    phase.requires_grad = True

    optimizer = torch.optim.NAdam([phase], lr=learning_rate)

    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()

        # the norm mode here should be the same as the FFT norm used when calculating the average_magnitude.
        x_n = fourier_to_image(unnormalized_average_magnitude, phase)[None, :, :, :]
        normalized_x_n = normalize_transform(x_n)

        # NOTE: spectrum shape might be different from model_input_shape.
        H, W = model_input_shape  # noqa: N806
        cropped_x_n = NormalRandomResizedCrop(mean=0.25, std=0.1, resize_to=(H, W))(
            normalized_x_n
        )

        noise_std = noise_stds[i]
        cropped_x_n += torch.randn_like(cropped_x_n) * noise_std
        cropped_x_n += torch.rand_like(cropped_x_n) * noise_std - (noise_std / 2.0)

        logits = model(cropped_x_n)
        target_logit = logits[0, target_logit_idx]

        loss = -target_logit
        loss.backward(retain_graph=True)

        optimizer.step()

    return phase.cpu().detach()


if __name__ == "__main__":
    import pathlib

    import timm
    import torchvision.datasets as datasets
    import torchvision.utils as vutils
    from timm.data.transforms import MaybeToTensor
    from torch.utils.data import DataLoader
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unnormalized_average_magnitude_path = pathlib.Path(
        "outputs/unnormalized_average_magnitude.pth"
    )
    if unnormalized_average_magnitude_path.exists():
        unnormalized_average_magnitude = torch.load(unnormalized_average_magnitude_path)
    else:
        transform = Compose(
            [
                Resize(
                    248,
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                CenterCrop((224, 224)),
                MaybeToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(
            root=pathlib.Path("data/imagenet/val"), transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        unnormalized_average_magnitude = compute_unnormalized_average_magnitude(
            dataloader, device=device
        )
        torch.save(unnormalized_average_magnitude, unnormalized_average_magnitude_path)

    # upscale the unnormalized_average_magnitude.
    unnormalized_average_magnitude = unnormalized_average_magnitude.unsqueeze(0)
    unnormalized_average_magnitude = torch.nn.functional.interpolate(
        unnormalized_average_magnitude,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    unnormalized_average_magnitude = unnormalized_average_magnitude.squeeze(0)

    print(
        f"unnormalized_average_magnitude.shape: {unnormalized_average_magnitude.shape}"
    )

    # model = timm.create_model(
    #     "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    # )
    # model = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    model = timm.create_model("resnet50.a1_in1k", pretrained=True)
    model = model.eval()

    transform = Compose(
        [
            Normalize(
                mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])
            )
        ]
    )

    phase = run_maco(
        model,
        device,
        unnormalized_average_magnitude,
        transform,
        target_logit_idx=907,
        num_steps=1024,
        learning_rate=1.0,
    )

    x = fourier_to_image(unnormalized_average_magnitude, phase)
    print(torch.min(x), torch.max(x))
    vutils.save_image(x, "outputs/maco_result.png")
