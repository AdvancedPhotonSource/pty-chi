# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import numpy as np
from typing import Optional
import h5py
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from .settings import MovieFileTypes


dataset_name = "frames"


def append_array_to_h5(array: np.ndarray, file_path: str, create_new_file: bool = False):
    if not os.path.exists(file_path) or create_new_file:
        # Create new file and 3D dataset
        with h5py.File(file_path, "w") as f:
            max_shape = (
                None,
                array.shape[0],
                array.shape[1],
            )  # Unlimited depth, fixed height & width
            dset = f.create_dataset(
                dataset_name,
                shape=(1, array.shape[0], array.shape[1]),  # Initial shape (depth=1)
                maxshape=max_shape,
                dtype=array.dtype,
            )
            dset[0] = array  # Store the first array
    else:
        # Append data to the existing 3D dataset
        with h5py.File(file_path, "a") as f:
            dset = f[dataset_name]
            new_depth = dset.shape[0] + 1  # Increase depth by 1
            dset.resize((new_depth, array.shape[0], array.shape[1]))  # Expand along depth
            dset[-1] = array  # Append new 2D array at the last depth


def save_movie_to_file(
    array: np.ndarray,
    file_type: MovieFileTypes,
    output_path: str,
    fps: int = 30,
    colormap: int = cv2.COLORMAP_BONE,
    enhance_contrast: bool = False,
    titles: list[str] = None,
    compress: bool = True,
    upper_bound: Optional[float] = None,
    lower_bound: Optional[float] = None,
):
    if upper_bound is not None:
        array = np.clip(array, a_max=upper_bound, a_min=None)
    if lower_bound is not None:
        array = np.clip(array, a_min=lower_bound, a_max=None)

    if file_type == MovieFileTypes.GIF:
        numpy_to_gif(array, output_path, fps, colormap, enhance_contrast, titles, compress=compress)
    elif file_type == MovieFileTypes.MP4:
        numpy_to_mp4(array, output_path, fps, colormap, enhance_contrast)


def numpy_to_gif(
    array: np.ndarray,
    output_path: str,
    fps: int = 30,
    colormap: int = cv2.COLORMAP_BONE,
    enhance_contrast: bool = False,
    titles: list[str] = None,
    font_size: int = 20,
    compress: bool = True,
):
    """
    Convert a 3D NumPy array (frames along the first axis) to a GIF file.

    Parameters:
        array (np.ndarray): 3D NumPy array of shape (frames, height, width)
        output_path (str): Path to save the GIF file.
        fps (int): Frames per second for the output video.
        colormap (int): OpenCV colormap to apply.
        enhance_contrast (bool): Whether to apply contrast enhancement (default: False).
    """
    frames, height, width = array.shape[:3]
    is_grayscale = len(array.shape) == 3
    gif_frames = []

    if titles and len(titles) != frames:
        raise ValueError("Length of titles list must match the number of frames.")
    
    # Apply contrast enhancement if needed
    if enhance_contrast:
        array = np.power(array, 0.5)  # Apply gamma correction to boost small values

    # Normalize to [0, 255] and convert to uint8
    array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    for i in range(frames):
        frame = array[i]

        if is_grayscale:
            frame = cv2.applyColorMap(frame, colormap)  # Apply colormap to grayscale frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL

        pil_frame = Image.fromarray(frame)

        # Draw title if provided
        if titles:
            draw = ImageDraw.Draw(pil_frame)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            text = titles[i]
            text_bbox = draw.textbbox((0, 0), text, font=font)  # Get text bounding box
            text_width = text_bbox[2] - text_bbox[0]  # Width of the text
            # text_height = text_bbox[3] - text_bbox[1]  # Height of the text
            position = ((width - text_width) // 2, 10)  # Centered at the top
            draw.text(position, text, font=font, fill=(255, 255, 255))

        gif_frames.append(pil_frame)

    # Save frames as GIF
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=compress,
    )
    print(f"GIF saved to {output_path}")


def numpy_to_mp4(
    array: np.ndarray,
    output_path: str,
    fps: int = 30,
    colormap: int = cv2.COLORMAP_BONE,
    enhance_contrast: bool = False,
):
    """
    Convert a 3D NumPy array (frames along the first axis) to an MP4 file.

    Parameters:
        array (np.ndarray): 3D NumPy array of shape (frames, height, width).
        output_path (str): Path to save the MP4 file.
        fps (int): Frames per second for the output video.
        colormap (int): OpenCV colormap to apply.
        enhance_contrast (bool): Whether to apply contrast enhancement (default: False).
    """
    frames, height, width = array.shape[:3]
    is_grayscale = len(array.shape) == 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(frames):
        frame = array[i]

        # Apply contrast enhancement if needed
        if enhance_contrast:
            frame = np.power(frame, 0.5)  # Apply gamma correction to boost small values

        # Normalize to [0, 255] and convert to uint8
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if is_grayscale:
            frame = cv2.applyColorMap(frame, colormap)  # Apply colormap to grayscale frame

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")
