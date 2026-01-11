import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from tbml.experiments.honeycomb.model import ConViT, ConViTConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize patch embeddings with PCA for a LeJEPA ConViT checkpoint."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument("image", type=str, help="Path to input image.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for the crop.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save side-by-side image.")
    return parser.parse_args()


def _dtype_from_name(name: str) -> jnp.dtype:
    """Map a dtype name to a JAX dtype.

    :param name: Dtype name string.
    :returns: JAX dtype.
    """
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve a checkpoint directory from a user-provided path.

    :param path: Path to a checkpoint directory or a checkpoint file.
    :returns: Resolved checkpoint directory.
    """
    if os.path.isdir(path) is True:
        ckpt_dir = path
    else:
        ckpt_dir = os.path.dirname(path)
        if ckpt_dir == "":
            raise ValueError("checkpoint path must be a directory or a file inside a directory")

    model_path = os.path.join(ckpt_dir, "model.eqx")
    metadata_path = os.path.join(ckpt_dir, "metadata.json")
    if os.path.isfile(model_path) is False:
        raise FileNotFoundError(f"model.eqx not found in checkpoint directory: {ckpt_dir}")
    if os.path.isfile(metadata_path) is False:
        raise FileNotFoundError(f"metadata.json not found in checkpoint directory: {ckpt_dir}")
    return ckpt_dir


def _load_run_config(checkpoint_dir: str) -> dict[str, Any]:
    """Load the training run configuration from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :returns: Run configuration dictionary.
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata.get("config")
    if config is None:
        raise ValueError("metadata.json does not contain a config entry")
    if isinstance(config, dict) is False:
        raise ValueError("metadata config entry must be a dictionary")
    return config


def _load_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    attn_implementation: str | None,
    model_config: dict[str, Any],
) -> ConViT:
    """Load a ConViT model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute and parameter dtype.
    :param attn_implementation: Attention backend implementation.
    :param model_config: Model configuration dictionary.
    :returns: Deserialized ConViT model.
    """
    config = ConViTConfig(**model_config)
    model_key = jax.random.PRNGKey(0)
    model = ConViT(
        config,
        dtype=dtype,
        param_dtype=dtype,
        attn_implementation=attn_implementation,
        key=model_key,
    )
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    return eqx.tree_deserialise_leaves(model_path, model)


def _sanitize_attention_impl(attn_impl: str | None) -> str | None:
    """Disable cuDNN attention when no GPU is available.

    :param attn_impl: Attention implementation string from config.
    :returns: Sanitized attention implementation.
    """
    if attn_impl != "cudnn":
        return attn_impl
    has_gpu = any(device.platform == "gpu" for device in jax.devices())
    if has_gpu is False:
        return None
    return attn_impl


def _resize_short_side(image: Image.Image, target: int) -> Image.Image:
    """Resize an image so its short side matches the target size.

    :param image: Input image.
    :param target: Target short-side size.
    :returns: Resized image.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive dimensions")
    if width <= height:
        new_width = target
        new_height = int(round(height * (target / float(width))))
    else:
        new_height = target
        new_width = int(round(width * (target / float(height))))
    return image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)


def _random_crop(
    image: Image.Image, crop_size: int, rng: np.random.Generator
) -> Image.Image:
    """Select a random square crop from an image.

    :param image: Input image.
    :param crop_size: Crop side length.
    :param rng: Random generator for crop selection.
    :returns: Cropped image.
    """
    width, height = image.size
    if crop_size > width or crop_size > height:
        raise ValueError("crop_size must not exceed resized image dimensions")
    max_left = width - crop_size
    max_top = height - crop_size
    if max_left > 0:
        left = int(rng.integers(0, max_left + 1))
    else:
        left = 0
    if max_top > 0:
        top = int(rng.integers(0, max_top + 1))
    else:
        top = 0
    return image.crop((left, top, left + crop_size, top + crop_size))


def _prepare_input(
    image: Image.Image,
    *,
    mean_std: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
) -> np.ndarray:
    """Convert a PIL image to a model input array with optional standardization.

    :param image: Cropped RGB image.
    :param mean_std: Optional mean and std tuples.
    :returns: Float32 array of shape (H, W, C).
    """
    array = np.asarray(image, dtype=np.float32)
    if mean_std is not None:
        mean, std = mean_std
        array = array / np.float32(255.0)
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        array = (array - mean_arr[None, None, :]) / std_arr[None, None, :]
    return array


def _compute_patch_embeddings(
    model: ConViT,
    image_array: np.ndarray,
    *,
    dtype: jnp.dtype,
) -> np.ndarray:
    """Compute patch embeddings before final normalization.

    :param model: ConViT model.
    :param image_array: Image array of shape (H, W, C).
    :param dtype: Compute dtype for the model input.
    :returns: Patch embeddings of shape (num_patches, dim).
    """
    if image_array.ndim != 3:
        raise ValueError("image_array must have shape (H, W, C)")
    image_tensor = jnp.asarray(image_array, dtype=dtype)
    image_tensor = jnp.expand_dims(image_tensor, axis=0)
    embeddings = model.encode_patches(image_tensor, train=False, key=None)
    embeddings = jax.device_get(embeddings)
    patches = np.asarray(embeddings, dtype=np.float32)[0]
    if model.config.use_cls_token is True:
        if patches.shape[0] <= 0:
            raise ValueError("expected at least one patch embedding")
        patches = patches[:-1]
    return patches


def _pca_to_rgb(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 3D via PCA and scale to RGB.

    :param embeddings: Patch embeddings of shape (num_patches, dim).
    :returns: Array of shape (num_patches, 3) scaled to [0, 255].
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (num_patches, dim)")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    components = v_t[:3, :]
    scores = centered @ components.T

    colors = np.empty_like(scores, dtype=np.float32)
    for idx in range(3):
        channel = scores[:, idx]
        min_val = float(channel.min())
        max_val = float(channel.max())
        if max_val > min_val:
            scaled = (channel - min_val) / (max_val - min_val)
        else:
            scaled = np.full(channel.shape, 0.5, dtype=np.float32)
        colors[:, idx] = scaled

    colors = np.clip(colors * 255.0, 0.0, 255.0)
    return colors.astype(np.uint8)


def _render_pca_image(
    colors: np.ndarray,
    *,
    grid_size: tuple[int, int],
    patch_size: int,
) -> Image.Image:
    """Render a PCA visualization image from per-patch RGB values.

    :param colors: Array of shape (num_patches, 3) with uint8 RGB values.
    :param grid_size: Patch grid size as (height, width).
    :param patch_size: Patch side length.
    :returns: PIL image of the PCA visualization.
    """
    grid_h, grid_w = grid_size
    num_patches = grid_h * grid_w
    if colors.shape != (num_patches, 3):
        raise ValueError("colors must match the number of patches")

    small = colors.reshape((grid_h, grid_w, 3))
    small_image = Image.fromarray(small, mode="RGB")
    target_size = (grid_w * patch_size, grid_h * patch_size)
    return small_image.resize(target_size, resample=Image.Resampling.BILINEAR)


def _combine_images(left: Image.Image, right: Image.Image) -> Image.Image:
    """Combine two images side-by-side.

    :param left: Left image.
    :param right: Right image.
    :returns: Combined image.
    """
    if left.size != right.size:
        raise ValueError("images must have the same size for side-by-side display")
    width, height = left.size
    combined = Image.new("RGB", (width * 2, height))
    combined.paste(left, (0, 0))
    combined.paste(right, (width, 0))
    return combined


def main() -> None:
    """Run the patch embedding PCA visualization."""
    args = _parse_args()
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    config = _load_run_config(checkpoint_dir)

    model_config = config.get("model")
    if model_config is None or isinstance(model_config, dict) is False:
        raise ValueError("run config missing model configuration")
    attention_config = config.get("attention")
    if attention_config is None or isinstance(attention_config, dict) is False:
        raise ValueError("run config missing attention configuration")
    training_config = config.get("training")
    if training_config is None or isinstance(training_config, dict) is False:
        raise ValueError("run config missing training configuration")
    data_config = config.get("data")
    if data_config is None or isinstance(data_config, dict) is False:
        raise ValueError("run config missing data configuration")

    dtype_name = training_config.get("dtype")
    if dtype_name is None or isinstance(dtype_name, str) is False:
        raise ValueError("training config missing dtype string")
    dtype = _dtype_from_name(dtype_name)
    attn_impl = attention_config.get("implementation")
    if attn_impl is not None and isinstance(attn_impl, str) is False:
        raise ValueError("attention implementation must be a string or null")
    attn_impl = _sanitize_attention_impl(attn_impl)

    model = _load_model(
        checkpoint_dir,
        dtype=dtype,
        attn_implementation=attn_impl,
        model_config=model_config,
    )

    image_path = args.image
    if os.path.isfile(image_path) is False:
        raise FileNotFoundError(f"image not found: {image_path}")
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = _resize_short_side(image, 256)
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()
    crop = _random_crop(resized, 224, rng)

    mean_std = data_config.get("mean_std")
    mean_std_tuple: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    if mean_std is None:
        mean_std_tuple = None
    else:
        if (
            isinstance(mean_std, list) is False
            or len(mean_std) != 2
            or isinstance(mean_std[0], list) is False
            or isinstance(mean_std[1], list) is False
        ):
            raise ValueError("mean_std must be a pair of lists")
        mean_std_tuple = (
            (float(mean_std[0][0]), float(mean_std[0][1]), float(mean_std[0][2])),
            (float(mean_std[1][0]), float(mean_std[1][1]), float(mean_std[1][2])),
        )

    input_array = _prepare_input(crop, mean_std=mean_std_tuple)
    embeddings = _compute_patch_embeddings(model, input_array, dtype=dtype)
    colors = _pca_to_rgb(embeddings)

    grid_size = model.patch_embed.grid_size
    pca_image = _render_pca_image(colors, grid_size=grid_size, patch_size=model.config.patch_size)
    combined = _combine_images(crop, pca_image)

    if args.output is not None:
        combined.save(args.output)
    combined.show()


if __name__ == "__main__":
    main()
