from pathlib import Path
import re
import sys
import numpy as np
import vtracer
from tqdm import tqdm


def save_to_svg(frames: list[np.ndarray], video_path: Path):
    """Saves a list of frames as a differentially rendered, animated SVG."""
    print(f"Saving to SVG using differential rendering: {video_path}")
    if not frames:
        print("Warning: No frames to save for SVG.", file=sys.stderr)
        return

    height, width, _ = frames[0].shape
    fps = 60.0
    delay_s = 1.0 / fps

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        # Use a clip path to prevent any potential overflow from vectorization artifacts.
        f'<defs><clipPath id="clip"><rect x="0" y="0" width="{width}" height="{height}"/></clipPath></defs>',
        # Apply the clip path to a main container group.
        '<g style="clip-path: url(#clip);">',
    ]

    # --- Vectorize and add the first frame (base frame) ---
    print("Vectorizing base frame...")
    base_frame = frames[0]
    rgba_frame = np.dstack((base_frame, np.full((height, width), 255, dtype=np.uint8)))
    flat_rgba = rgba_frame.reshape(-1, 4)
    rgba_pixels = [tuple(pixel) for pixel in flat_rgba]

    # Use polygon mode for blocky Atari graphics and disable filters for max detail.
    svg_string = vtracer.convert_pixels_to_svg(
        rgba_pixels, (width, height), mode="none", filter_speckle=0, length_threshold=0.0
    )

    match = re.search(r"<svg[^>]*>(.*)</svg>", svg_string, re.DOTALL)
    if match:
        # The base frame is always visible.
        svg_parts.append(f'<g visibility="visible">{match.group(1)}</g>')

    # --- Vectorize and add differential frames ---
    print("Vectorizing differential frames...")
    prev_frame = base_frame
    for i, curr_frame in tqdm(enumerate(frames[1:], start=1), total=len(frames) - 1, desc="Processing Diffs"):
        # Find differences between current and previous frame.
        diff_mask = (prev_frame != curr_frame).any(axis=2)

        # If there's no change, we don't need to add a layer.
        if not np.any(diff_mask):
            prev_frame = curr_frame
            continue

        # Create a transparent frame, copying only the changed pixels.
        diff_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        diff_rgba[diff_mask, :3] = curr_frame[diff_mask]  # Copy RGB from current frame.
        diff_rgba[diff_mask, 3] = 255  # Set Alpha to opaque for changed pixels.

        # Vectorize the sparse differential frame.
        flat_diff_rgba = diff_rgba.reshape(-1, 4)
        diff_pixels = [tuple(pixel) for pixel in flat_diff_rgba]
        diff_svg_string = vtracer.convert_pixels_to_svg(
            diff_pixels, (width, height), mode="none", filter_speckle=0, length_threshold=0.0
        )

        match = re.search(r"<svg[^>]*>(.*)</svg>", diff_svg_string, re.DOTALL)
        if match:
            svg_content = match.group(1)
            if svg_content.strip():  # Only add a group if there's vector content.
                begin_time = i * delay_s
                svg_parts.append('<g visibility="hidden">')
                svg_parts.append(f'  <set attributeName="visibility" to="visible" begin="{begin_time:.4f}s" />')
                svg_parts.append(svg_content)
                svg_parts.append("</g>")

        prev_frame = curr_frame

    svg_parts.append("</g>")  # Close the main container group.
    svg_parts.append("</svg>")  # Close the svg.

    print("Writing SVG file...")
    with video_path.open("w", encoding="utf-8") as f:
        f.write("".join(svg_parts))

    print(f"Differentially rendered SVG animation saved successfully to {video_path}")