from typing import Literal

import torch
import torchvision
from tqdm import tqdm


class NormalRandomResizedCrop:
    """Randomly crops and optionally resizes an image using a normal distribution for the crop ratio.

    This class expectes to perform like `torchvision.transforms.RandomResizedCrop`.
    https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html

    """

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
            image (torch.Tensor): shape (N, C, H, W)
                The input image tensor.

        Returns:
            torch.Tensor: shape (N, C, H, W)
                The cropped (and possibly resized) image tensor. The output shape will match the input dimensions.

        """
        assert image.ndim == 4

        N, _, H, W = image.shape  # noqa: N806
        device = image.device

        # sample the crop ratio from a normal distribution N(mean, std) and clamp between 0.1 and 1.0.
        crop_frac = torch.normal(mean=self.mean, std=self.std, size=(N,), device=device)
        crop_frac = crop_frac.clamp(0.1, 1.0)

        # crop sizes of each image in the batch.
        crop_h = (H * crop_frac).floor().to(torch.int64)  # shape: (N,)
        crop_w = (W * crop_frac).floor().to(torch.int64)  # shape: (N,)

        # choose a random crop position ensuring the crop is within image bounds.
        max_top = (H - crop_h).clamp(min=0)
        max_left = (W - crop_w).clamp(min=0)
        top = (
            (torch.rand(N, device=device) * (max_top + 1).to(torch.float32))
            .floor()
            .to(torch.int64)
        )
        left = (
            (torch.rand(N, device=device) * (max_left + 1).to(torch.float32))
            .floor()
            .to(torch.int64)
        )

        H_out, W_out = self.resize_to  # noqa: N806
        v = torch.linspace(0, 1, steps=H_out, device=device).view(
            1, H_out, 1
        )  # shape: (1, H_out, 1)
        u = torch.linspace(0, 1, steps=W_out, device=device).view(
            1, 1, W_out
        )  # shape: (1, 1, W_out)

        crop_h = crop_h.view(N, 1, 1).to(torch.float32)
        crop_w = crop_w.view(N, 1, 1).to(torch.float32)
        top = top.view(N, 1, 1).to(torch.float32)
        left = left.view(N, 1, 1).to(torch.float32)

        y_grid = top + v * (crop_h - 1)
        x_grid = left + u * (crop_w - 1)
        x_grid, y_grid = torch.broadcast_tensors(x_grid, y_grid)

        # normalize the grid to [-1, 1] for torch.nn.functional.grid_sample.
        y_grid_norm = (y_grid / (H - 1)) * 2 - 1
        x_grid_norm = (x_grid / (W - 1)) * 2 - 1

        grid = torch.stack(
            [x_grid_norm, y_grid_norm], dim=-1
        )  # shape: (N, H_out, W_out, 2)

        return torch.nn.functional.grid_sample(
            image, grid, mode="bilinear", align_corners=True
        )


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


def normalize_alpha(
    alpha: torch.Tensor,
    percentile: float = 80.0,
) -> torch.Tensor:
    """Take mean, clamp and normalize alpha.

    This function computes the mean of a 3-channel alpha tensor over its channel dimension,
    clamps the resulting single-channel tensor by the value at the specified percentile to eliminate extreme values,
    and normalizes the clamped values to the range [0, 1].

    Args:
        alpha (torch.Tensor): shape (3, H, W)
            A tensor representing an alpha with 3 channels.
        percentile (float, optional): The percentile (between 0.0 and 100.0) used to compute the clamping threshold.
            Values above this threshold are clamped. Defaults to 80.0.

    Returns:
        torch.Tensor: shape (1, H, W)
            A normalized tensor with values scaled to the range [0, 1].

    Raises:
        AssertionError: If the input tensor does not have 3 dimensions,
                        if the first dimension is not of size 3, or
                        if the percentile is not between 0.0 and 100.0.

    """
    assert len(alpha.shape) == 3
    assert alpha.size(0) == 3
    assert 0.0 <= percentile <= 100.0

    alpha_mean = torch.mean(alpha, dim=0, keepdim=True)  # (1, H, W)

    # clamp by the value at the 80th percentile to eliminate extreme values.
    alpha_clamped = torch.clamp(
        alpha_mean, max=torch.quantile(alpha_mean, percentile / 100.0)
    )

    # normalize the alpha to [0, 1].
    alpha_normalized = alpha_clamped / (alpha_clamped.max() + 1e-8)

    return alpha_normalized


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize the input tensor to the range [0, 1].

    This function subtracts the minimum value from the tensor and divides by the range
    (max - min) plus a small epsilon for numerical stability, effectively scaling
    the tensor values to be between 0 and 1.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        eps (float, optional): A small value added to the denominator to prevent division
            by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: A tensor with values normalized to the range [0, 1].

    """
    return (x - x.min()) / (x.max() - x.min() + eps)


def run_maco(
    model: torch.nn.Module,
    device: torch.device,
    unnormalized_average_magnitude: torch.Tensor,
    normalize_transform: torch.nn.Module,
    target_logit_idx: int = 0,
    num_steps: int = 256,
    num_crops: int = 32,
    noise_std: float = -1.0,
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
        num_crops (int, optional): Number of random crops to generate per image. Defaults to 32.
        noise_std (float, optional): Standard deviation of the noise added to the cropped images.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1.0.
        model_input_shape (tuple[int, int]): The input shape of the model. Defaults to (224, 224).

    Returns:
        torch.Tensor: The optimized phase tensor.

    """
    assert len(unnormalized_average_magnitude.shape) == 3
    assert unnormalized_average_magnitude.size(0) == 3
    assert len(model_input_shape) == 2

    model = model.to(device)
    unnormalized_average_magnitude = unnormalized_average_magnitude.to(device)

    if noise_std == -1.0:
        noise_stds = torch.logspace(0, -4, steps=num_steps, dtype=torch.float32)
        get_noise_std = lambda i: noise_stds[i]  # noqa: E731
    elif isinstance(noise_std, float):
        assert noise_std > 0.0
        get_noise_std = lambda _: noise_std  # noqa: E731
    else:
        raise ValueError(f"Invalid noise_std argument: {noise_std}")

    phase = (
        2 * torch.pi * torch.rand(unnormalized_average_magnitude.shape, device=device)
    ) - torch.pi
    phase.requires_grad = True

    alpha = torch.zeros(unnormalized_average_magnitude.shape, device=device)

    optimizer = torch.optim.NAdam([phase], lr=learning_rate)

    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()

        # the norm mode here should be the same as the FFT norm used when calculating the average_magnitude.
        x_n = fourier_to_image(unnormalized_average_magnitude, phase)[None, :, :, :]
        normalized_x_n = normalize_transform(x_n)
        normalized_x_n.requires_grad_(True)
        normalized_x_n.retain_grad()

        # NOTE: spectrum shape might be different from model_input_shape.
        cropped_x_n = NormalRandomResizedCrop(
            mean=0.25, std=0.1, resize_to=model_input_shape
        )(normalized_x_n.repeat(num_crops, 1, 1, 1))

        noise_std = get_noise_std(i)  # type: ignore[no-untyped-call]
        cropped_x_n += torch.randn_like(cropped_x_n) * noise_std
        cropped_x_n += torch.rand_like(cropped_x_n) * noise_std - (noise_std / 2.0)

        logits = model(cropped_x_n)
        target_logit = logits[:, target_logit_idx]

        loss = -target_logit.mean()
        loss.backward(retain_graph=True)

        grad_x_n = normalized_x_n.grad.squeeze(0)  # shape: (3, H, W)
        alpha += grad_x_n.abs()

        optimizer.step()

    return phase.cpu().detach(), alpha.cpu().detach()


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

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    )
    # model = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    # model = timm.create_model("resnet50.a1_in1k", pretrained=True)
    model = model.eval()

    transform = Compose(
        [
            Normalize(
                mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])
            )
        ]
    )

    phase, alpha = run_maco(
        model,
        device,
        unnormalized_average_magnitude,
        transform,
        target_logit_idx=1,
        num_steps=256,
        num_crops=32,
        learning_rate=1.0,
    )

    x = fourier_to_image(unnormalized_average_magnitude, phase)
    print(torch.min(x), torch.max(x))

    alpha_final = torch.mean(alpha, dim=0, keepdim=True)
    alpha_thresh = torch.quantile(alpha_final, 80.0 / 100.0)
    alpha_final = torch.clamp(alpha_final, max=alpha_thresh)
    alpha_final = alpha_final / (alpha_final.max() + 1e-8)  # (1, H, W)
    alpha_final = alpha_final.repeat(3, 1, 1)  # (3, H, W)

    x_final = x * alpha_final
    x_final = normalize(x_final)

    vutils.save_image([x, alpha_final, x_final], "outputs/maco_result.png")
