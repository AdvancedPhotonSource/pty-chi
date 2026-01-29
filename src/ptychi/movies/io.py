# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import numpy as np
from typing import Optional, Union
import h5py
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .settings import MovieFileTypes, ObjectMovieSettings, PlotTypes, ProbeMovieSettings, PositionsMovieSettings

movie_setting_types = Union[ObjectMovieSettings, ProbeMovieSettings, PositionsMovieSettings]
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
    image_type: PlotTypes = PlotTypes.IMAGE,
):
    if upper_bound is not None:
        array = np.clip(array, a_max=upper_bound, a_min=None)
    if lower_bound is not None:
        array = np.clip(array, a_min=lower_bound, a_max=None)

    if file_type == MovieFileTypes.GIF:
        if image_type == PlotTypes.IMAGE:
            numpy_to_gif(array, output_path, fps, colormap, enhance_contrast, titles, compress=compress)
        elif image_type == PlotTypes.LINE_PLOT:
            numpy_to_line_plot_gif(array, output_path, fps, titles=titles, marker=".")
    elif file_type == MovieFileTypes.MP4:
        if image_type == PlotTypes.IMAGE:
            numpy_to_mp4(array, output_path, fps, colormap, enhance_contrast, titles)
        elif image_type == PlotTypes.LINE_PLOT:
            numpy_to_line_plot_mp4(array, output_path, fps, titles=titles, marker=".")


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
    titles: list[str] = None,
    font_size: int = 20,
):
    """
    Convert a 3D NumPy array (frames along the first axis) to an MP4 file.

    Parameters:
        array (np.ndarray): 3D NumPy array of shape (frames, height, width).
        output_path (str): Path to save the MP4 file.
        fps (int): Frames per second for the output video.
        colormap (int): OpenCV colormap to apply.
        enhance_contrast (bool): Whether to apply contrast enhancement (default: False).
        titles (list[str]): List of titles for each frame (optional).
        font_size (int): Font size for titles (default: 20).
    """
    frames, height, width = array.shape[:3]
    is_grayscale = len(array.shape) == 3

    if titles and len(titles) != frames:
        raise ValueError("Length of titles list must match the number of frames.")

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

        # Add title if provided
        if titles:
            text = titles[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_size / 30.0  # Scale font size appropriately
            color = (255, 255, 255)  # White text
            thickness = max(1, int(font_scale * 2))  # Adjust thickness based on font scale
            
            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = text_height + 10  # Position near the top with some padding
            
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")


def numpy_to_line_plot_gif(
    array: np.ndarray,
    filename: str,
    fps: int = 10,
    dpi: int = 100,
    titles: list[str] = None,
    color: str = "blue",
    linewidth: float = 1.0,
    marker: Optional[str] = None,
    markersize: float = 3.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    grid: bool = True,
    figsize: tuple = (8, 6),
) -> None:
    """
    Convert a 3D numpy array to an animated GIF using line plots.
    
    For each frame i, plots array[i][:, 1] vs array[i][:, 0].

    Parameters
    ----------
    array : np.ndarray
        3D array with shape (frames, n_points, 2) where the last dimension
        contains [x, y] coordinates for plotting.
    filename : str
        Output filename for the GIF.
    fps : int, optional
        Frames per second for the animation, by default 10.
    dpi : int, optional
        Resolution of the output GIF, by default 100.
    color : str, optional
        Color of the line plot, by default "blue".
    linewidth : float, optional
        Width of the line, by default 1.0.
    marker : str, optional
        Marker style for data points, by default None.
    markersize : float, optional
        Size of markers, by default 3.0.
    title : str, optional
        Title for the plot, by default None.
    xlabel : str, optional
        Label for x-axis, by default None.
    ylabel : str, optional
        Label for y-axis, by default None.
    xlim : tuple, optional
        X-axis limits as (min, max), by default None (auto-scale).
    ylim : tuple, optional
        Y-axis limits as (min, max), by default None (auto-scale).
    grid : bool, optional
        Whether to show grid, by default True.
    figsize : tuple, optional
        Figure size as (width, height), by default (8, 6).
    """
    if array.ndim != 3:
        raise ValueError("Array must be 3-dimensional with shape (frames, n_points, 2)")
    
    if array.shape[2] != 2:
        raise ValueError("Last dimension of array must be 2 (for x, y coordinates)")
    
    frames, n_points, _ = array.shape
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Determine axis limits if not provided
    if xlim is None:
        x_min = np.min(array[:, :, 1])
        x_max = np.max(array[:, :, 1])
        x_range = x_max - x_min
        xlim = (x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    
    if ylim is None:
        y_min = np.min(array[:, :, 0])
        y_max = np.max(array[:, :, 0])
        y_range = y_max - y_min
        ylim = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Set up the plot
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Initialize empty line
    line, = ax.plot([], [], color=color, linewidth=linewidth, 
                   marker=marker, markersize=markersize)
    
    # Initialize frame number text if enabled
    frame_text = None
    if titles is not None:
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame_idx):
        """Animation function for each frame."""
        x_data = array[frame_idx][:, 1]
        y_data = array[frame_idx][:, 0]
        line.set_data(x_data, y_data)
        
        # Update frame number text if enabled
        # if show_frame_number and frame_text is not None:
        if titles is not None:
            # frame_text.set_text(frame_number_format.format(frame=frame_idx))
            frame_text.set_text(str(titles[frame_idx]))
            return line, frame_text
        
        return line,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1000/fps, blit=True, repeat=True
    )
    
    # Save as GIF
    anim.save(filename, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Line plot GIF saved to {filename}")


def numpy_to_line_plot_mp4(
    array: np.ndarray,
    filename: str,
    fps: int = 10,
    dpi: int = 100,
    titles: list[str] = None,
    color: str = "blue",
    linewidth: float = 1.0,
    marker: Optional[str] = None,
    markersize: float = 3.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    grid: bool = True,
    figsize: tuple = (8, 6),
) -> None:
    """
    Convert a 3D numpy array to an animated MP4 using line plots.
    
    For each frame i, plots array[i][:, 1] vs array[i][:, 0].

    Parameters
    ----------
    array : np.ndarray
        3D array with shape (frames, n_points, 2) where the last dimension
        contains [x, y] coordinates for plotting.
    filename : str
        Output filename for the MP4.
    fps : int, optional
        Frames per second for the animation, by default 10.
    dpi : int, optional
        Resolution of the output MP4, by default 100.
    color : str, optional
        Color of the line plot, by default "blue".
    linewidth : float, optional
        Width of the line, by default 1.0.
    marker : str, optional
        Marker style for data points, by default None.
    markersize : float, optional
        Size of markers, by default 3.0.
    title : str, optional
        Title for the plot, by default None.
    xlabel : str, optional
        Label for x-axis, by default None.
    ylabel : str, optional
        Label for y-axis, by default None.
    xlim : tuple, optional
        X-axis limits as (min, max), by default None (auto-scale).
    ylim : tuple, optional
        Y-axis limits as (min, max), by default None (auto-scale).
    grid : bool, optional
        Whether to show grid, by default True.
    figsize : tuple, optional
        Figure size as (width, height), by default (8, 6).
    """
    if array.ndim != 3:
        raise ValueError("Array must be 3-dimensional with shape (frames, n_points, 2)")
    
    if array.shape[2] != 2:
        raise ValueError("Last dimension of array must be 2 (for x, y coordinates)")
    
    frames, n_points, _ = array.shape
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Determine axis limits if not provided
    if xlim is None:
        x_min = np.min(array[:, :, 1])
        x_max = np.max(array[:, :, 1])
        x_range = x_max - x_min
        xlim = (x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    
    if ylim is None:
        y_min = np.min(array[:, :, 0])
        y_max = np.max(array[:, :, 0])
        y_range = y_max - y_min
        ylim = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # Set up the plot
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Initialize empty line
    line, = ax.plot([], [], color=color, linewidth=linewidth, 
                   marker=marker, markersize=markersize)
    
    # Initialize frame number text if enabled
    frame_text = None
    if titles is not None:
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame_idx):
        """Animation function for each frame."""
        x_data = array[frame_idx][:, 1]
        y_data = array[frame_idx][:, 0]
        line.set_data(x_data, y_data)
        
        # Update frame number text if enabled
        if titles is not None:
            frame_text.set_text(str(titles[frame_idx]))
            return line, frame_text
        
        return line,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1000/fps, blit=True, repeat=True
    )
    
    # Save as MP4
    anim.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Line plot MP4 saved to {filename}")
