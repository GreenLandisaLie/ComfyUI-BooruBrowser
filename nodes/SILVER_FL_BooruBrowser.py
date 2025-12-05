import re
import requests
import math
import random
import io
import json
import os
import subprocess
import tempfile
import shutil
import uuid
from PIL import Image, ImageOps
from typing import Optional, Dict, Any
import torch
import numpy as np
import cv2
import aiohttp
from aiohttp import web
import asyncio
import boto3
import comfy
import folder_paths
from server import PromptServer


USER_AGENT = "ComfyUI-BooruBrowser/1.0"

CURRENT_URLS = []
RANDOMLY_SELECTED_URL = ""


# ----- VIDEO UTILS ------
TEMP_FOLDER = folder_paths.get_temp_directory()
os.makedirs(TEMP_FOLDER, exist_ok=True)

def check_ffmpeg():
    """
    Checks if the 'ffmpeg' executable is installed and available in PATH.
    """
    # 1. Check using shutil.which (Most efficient)
    if shutil.which("ffmpeg"):
        return True
    
    # 2. Check using subprocess (Fallback, ensures it's executable)
    try:
        # Run a minimal command, suppress output, check only the return code
        # The 'shell=True' option is generally avoided, so we execute directly.
        subprocess.run(
            ['ffmpeg', '-version'], 
            check=True,                 # Raise CalledProcessError for non-zero return codes
            stdout=subprocess.PIPE,     # Don't print stdout to console
            stderr=subprocess.PIPE,     # Don't print stderr to console
            text=True                   # Decode output as text
        )
        return True
    except FileNotFoundError:
        # This exception is raised if the executable 'ffmpeg' cannot be found in PATH.
        return False
    except subprocess.CalledProcessError:
        # This exception is raised if the command runs but returns a non-zero exit code
        # (e.g., if there's a problem with the FFMPEG installation or environment, 
        # though '-version' rarely fails).
        return True # The command ran, so FFMPEG is technically installed.
    except Exception:
        # Catch any other unexpected errors
        return False

FFMPEG_available = check_ffmpeg()

def run_ffprobe_on_url(file_url: str) -> Optional[dict]:
    """Runs ffprobe directly on file_url and returns JSON metadata."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_entries", "stream=tags,side_data_list,displaymatrix",
                "-show_format",
                "-show_streams",
                file_url
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if proc.returncode != 0:
            # ffprobe failed but sometimes stderr contains useful info
            print("ffprobe URL error:", proc.stderr)
            return None
        
        return json.loads(proc.stdout)
    
    except Exception as e:
        print(f"Failed to probe video: {e}")
        return None

def run_ffprobe_on_bytes(data: bytes, file_url: str) -> Optional[dict]:
    """Runs ffprobe on byte data fed via stdin and returns JSON metadata."""
    try:
        proc = subprocess.Popen(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_entries", "stream=tags,side_data_list,displaymatrix",
                "-show_streams",
                "pipe:0"
            ],            
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = proc.communicate(input=data, timeout=5)
        
        if proc.returncode != 0:
            return run_ffprobe_on_url(file_url)
        
        metadata = json.loads(out.decode("utf-8"))
        return metadata
    except Exception as e:
        print(f"Failed to probe video: {e}")
        return None

def probe_video(file_url):
    head = requests.head(file_url)
    if "Content-Length" not in head.headers:
        #print("Server did not provide Content-Length; abort.")
        return None
    
    MOOV_PROBE_SIZE = 1024 * 1024 * 1  # 1 MB from start & end
    content_length = int(head.headers["Content-Length"])
    ranges = [
        f"bytes=0-{MOOV_PROBE_SIZE - 1}",
        f"bytes={max(content_length - MOOV_PROBE_SIZE, 0)}-{content_length - 1}"
    ]
    
    probe_data = bytearray()
    for r in ranges:
        resp = requests.get(file_url, headers={"Range": r}, stream=True)
        if resp.status_code not in (200, 206):
            print(f"Failed to fetch probe range {r}")
            return None
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            probe_data.extend(chunk)
    
    metadata = run_ffprobe_on_bytes(bytes(probe_data), file_url)
    if not metadata:
        #print("FFprobe could not parse metadata.")
        return None
    return metadata

def get_video_duration_and_rotation(file_url):
    duration_ms = 0
    rotation = 0
    
    if not file_url or not FFMPEG_available:
        return duration_ms, rotation
    
    metadata = probe_video(file_url)
    if metadata:
        # === Duration parser ===
        if "format" in metadata and "duration" in metadata["format"]:
            try:
                duration_ms = int(float(metadata["format"]["duration"]) * 1000)
            except:
                pass
        if duration_ms == 0 and "streams" in metadata:
            for stream in metadata["streams"]:
                if "duration" in stream and float(stream["duration"]) > 0:
                    duration_ms = int(float(stream["duration"]) * 1000)
                    break
        
        # === Rotation parser ===
        # 1. Check simple rotation tags ("rotate" or "rotation")
        if "streams" in metadata and len(metadata["streams"]) > 0:
            first_stream = metadata["streams"][0]
            tags = first_stream.get("tags", {})
            for key in ("rotate", "rotation"):
                if key in tags:
                    try:
                        angle = int(tags[key])
                        rotation = abs(angle) % 360
                        break
                    except:
                        pass
            
            if rotation == 0:
                # 2. Check side_data_list for rotation or rotate fields
                sdl = first_stream.get("side_data_list", [])
                for sd in sdl:
                    if "rotation" in sd:
                        try:
                            rotation = abs(int(sd["rotation"])) % 360
                            break
                        except:
                            pass
                    if "rotate" in sd:
                        try:
                            rotation = abs(int(sd["rotate"])) % 360
                            break
                        except:
                            pass
                    
                    if rotation == 0:
                        # 3. Check displaymatrix
                        if "displaymatrix" in sd:
                            text = str(sd["displaymatrix"])
                            m = re.search(r'(-?\d+)', text)
                            if m:
                                try:
                                    return abs(int(m.group(1))) % 360
                                except:
                                    pass
    
    return duration_ms, rotation

def isVideo(file_url):
    return (
        file_url.endswith(".gif") or
        file_url.endswith(".gifv") or
        file_url.endswith(".webm") or
        file_url.endswith(".mp4") or
        file_url.endswith(".m4v") or
        file_url.endswith(".mov") or
        file_url.endswith(".ogv")
    )

def download_temp_clip(file_url: str, extract_time_start_ms: int, extract_duration_range_ms: int) -> Optional[str]:
    start_time_seconds = extract_time_start_ms / 1000.0
    duration_seconds = extract_duration_range_ms / 1000.0

    temp_file_path = os.path.join(TEMP_FOLDER, f"temp_clip_{uuid.uuid4()}.mp4")
    
    ffmpeg_command = [
        'ffmpeg',
        '-y',               # Overwrite output file
        '-i', file_url,
        '-ss', str(start_time_seconds),
        '-t', str(duration_seconds),
        '-an',              # No audio
        '-c:v', 'libx264',  # Re-encode video stream
        '-preset', 'fast',  # Balance between speed and compression (e.g., 'veryfast', 'medium')
        '-crf', '18',       # Constant Rate Factor: 0=lossless, 51=worst quality. 23 is a good default.
        '-f', 'mp4',        # Output format
        temp_file_path
    ]
    
    try:
        process = subprocess.run(
            ffmpeg_command,
            check=True,  # Raise an exception for non-zero return codes
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return temp_file_path
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        os.unlink(temp_file_path)
        return None

def download_temp_gif(file_url):
    try:
        #ext = ".gifv" if file_url.endswith(".gifv") else ".gif"
        ext = ".gif"
        temp_file_path = os.path.join(TEMP_FOLDER, f"temp_gif_{uuid.uuid4()}{ext}")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        with open(temp_file_path, 'wb') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
        return temp_file_path
    except:
        return None

def convert_gif_to_mp4(gif_path):
    try:
        mp4_path = os.path.join(TEMP_FOLDER, f"converted_video_{uuid.uuid4()}.mp4")
        ffmpeg_command = [
            'ffmpeg',
            '-i', gif_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', 'faststart',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-crf', '18',
            mp4_path
        ]
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        os.unlink(gif_path)
        return mp4_path
    except Exception as e:
        print(f"An unexpected error occurred during GIF to MP4 conversion: {e}")
        if os.path.exists(mp4_path):
            os.unlink(mp4_path)
        return None

def calculate_blurriness(frame):
    """
    FFT high-frequency energy.
    LOWER return value = SHARPER frame.
    """

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Downscale to reduce FFT cost (factor 2–3 is ideal) -- ONLY FOR IMAGES EQUAL OR HIGHER THAN 640x640
    #    This does NOT meaningfully affect sharpness measurement.
    height, width = frame.shape[:2]
    if height * width >= 640 * 640:
        img = cv2.resize(gray, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    else:
        img = gray
    
    # 3. Convert to float32
    img = img.astype(np.float32)
    
    # 4. FFT
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # 5. Compute magnitude spectrum
    mag = np.abs(Fshift)

    # 6. Define high-frequency mask:
    #    Keep only outer 40% of the spectrum → best for blur detection.
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # radius threshold (tunable; 0.4–0.5 works well)
    radius = 0.4 * min(cy, cx)

    # distance grid
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv*xv + yv*yv)

    # high frequency region: outside the radius
    high_freq_mask = dist > radius

    # 7. High-frequency energy
    hf_energy = mag[high_freq_mask].mean() + 1e-6

    # 8. Invert so LOWER = SHARPER
    return 1.0 / hf_energy

def select_single_frame(file_url, extract_time_start_ms):
    t = torch.zeros(1,1,1,3)

    is_gif = file_url.endswith(".gif")# or file_url.endswith(".gifv")
    if is_gif:
        file_url = download_temp_gif(file_url)
        try:
            img = Image.open(file_url)
            # Calculate the target frame index
            # GIFs usually run at a fixed frame delay (often 100ms), 
            # but the best way is to accumulate delays.
            # We'll approximate the index by iterating and summing frame durations
            current_duration_ms = 0
            target_index = 0
            while current_duration_ms < extract_time_start_ms:
                # Get the frame's specific duration (delay)
                # GIF frame delay is in centiseconds (1/100th of a second), so multiply by 10
                delay_ms = img.info.get('duration', 100) * 10
                # Move to the next frame
                try:
                    img.seek(target_index + 1)
                    current_duration_ms += delay_ms
                    target_index += 1
                except EOFError:
                    # Reached the end of the animation
                    break
            img.seek(max(0, target_index - 1)) # Seek back to the identified target frame. Use the last successful index
            t = torch.from_numpy(np.array(img.convert("RGB"), np.float32) / 255.0)  # H,W,C
            t = t.unsqueeze(0)
        except Exception as e:
            print(f"Error parsing gif: {e}")
        finally:
            if img:
                img.close()
            if file_url:
                os.unlink(file_url)
        return t
    
    cmd = [
        "ffmpeg",
        "-ss", f"{extract_time_start_ms / 1000.0}",
        "-i", file_url,
        "-copyts",
        "-vframes", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-"
    ]
    
    try:
        # Run the command and capture the raw PNG data from stdout
        raw_img_png_buffer = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        np_buffer = np.frombuffer(raw_img_png_buffer, np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR) # Decode the PNG format into a BGR image array (H, W, 3)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(image.astype(np.float32) / 255.0)  # H,W,C
            t = t.unsqueeze(0) # (1, H, W, C)
    except Exception as e:
        print(f"Error during image decoding/tensor conversion: {e}")
    
    return t

def extract_video_frames(file_url, extract_time_start_ms, extract_duration_range_ms, extract_N_frames, frame_max_width, frame_max_height, filter_blurry_frames):
    t = torch.zeros(1,1,1,3)
    
    is_gif = file_url.endswith(".gif")
    if is_gif:
        tmp_clip_path = download_temp_gif(file_url)
        try:
            tmp_clip_path = convert_gif_to_mp4(tmp_clip_path)
        except:
            print(f"Failed to parse gif from url: {file_url}")
            if tmp_clip_path:
                os.unlink(tmp_clip_path)
            return t
    else:
        tmp_clip_path = download_temp_clip(file_url, extract_time_start_ms, extract_duration_range_ms)
    
    if not tmp_clip_path:
        return t
    
    cap = cv2.VideoCapture(tmp_clip_path)
    if not cap.isOpened():
        os.unlink(tmp_clip_path)
        print("Failed to open temporary clipped video with OpenCV")
        return t

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        os.unlink(tmp_clip_path)
        print("Clipped video has no frames")
        return t
        
    _, required_rotation = get_video_duration_and_rotation(file_url)
    
    extract_N_frames = 1 if extract_N_frames <= 1 else extract_N_frames 
    if extract_N_frames == 1:
        base_target_idxs = [0]
    else:
        step = (total_frames - 1) / (extract_N_frames - 1)
        base_target_idxs = [min(int(round(i * step)), total_frames - 1) for i in range(extract_N_frames)]
    
    # Build ±N candidate windows (up to ±5 when enough total_frames)
    WINDOW_SIZE = 0
    for N in [5, 4, 3, 2, 1]:
        if total_frames >= extract_N_frames * ((N * 2) + 1):
            WINDOW_SIZE = N
            break
    
    def build_windows(base_idxs):
        windows = []
        for idx in base_idxs:
            s = max(0, idx - WINDOW_SIZE)
            e = min(total_frames - 1, idx + WINDOW_SIZE)
            windows.append(list(range(s, e + 1)))
        return windows

    frame_windows = build_windows(base_target_idxs)
    filter_blurry_frames = filter_blurry_frames and extract_N_frames > 1 and WINDOW_SIZE > 0 and not is_gif
    
    # 1. Determine final set of unique indices to read
    if filter_blurry_frames:
        unique_target_idxs = set() # Use a set to collect unique candidate indices
        for window in frame_windows:
            unique_target_idxs.update(window)
        # Sort them for better sequential reading performance from cv2
        target_idxs_to_read = sorted(list(unique_target_idxs))
    else:
        target_idxs_to_read = base_target_idxs
    
    frameIDX_frame = {}
    dimensions_checked = False
    new_width = 0
    new_height = 0

    # 2. Read frames by index (addresses memory bottleneck)
    for idx in target_idxs_to_read:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not dimensions_checked and ret and frame is not None:
            dimensions_checked = True
            (original_height, original_width) = frame.shape[:2]
            width_ratio = frame_max_width / original_width
            height_ratio = frame_max_height / original_height
            scale_factor = min(width_ratio, height_ratio)
            if scale_factor < 1.0:
                new_width = int(original_width * scale_factor) - (int(original_width * scale_factor) % 2)
                new_height = int(original_height * scale_factor) - (int(original_height * scale_factor) % 2)
        
        if not ret or frame is None:
            if len(frameIDX_frame) > 0:
                # Fallback: Copy last successfully selected frame
                last_key = list(frameIDX_frame.keys())[-1]
                frameIDX_frame[idx] = frameIDX_frame[last_key]
            # else: the index is truly missing, which ensure_dict_indices will handle
            continue
            
        frameIDX_frame[idx] = frame
    
    cap.release()
    os.unlink(tmp_clip_path)
    
    if len(frameIDX_frame) == 0:
        raise ValueError("No frames decoded.")
    
    # 3. Ensure all necessary indices exist (especially if first frame read failed)
    def ensure_dict_indices(d, indices):
        indices_in_d = list(d.keys())
        missing_indices = [idx for idx in indices if idx not in indices_in_d]
        
        for idx in missing_indices:
            # Find the closest index that *was* read successfully
            closest_idx = min(indices_in_d, key=lambda x: abs(x - idx))
            d[idx] = d[closest_idx]
            
        # Optimization: remove frames we read but aren't needed for the final selection (only relevant if filter_blurry_frames is False)
        unwanted_indices = [idx for idx in indices_in_d if idx not in indices] 
        for idx in unwanted_indices:
            del d[idx]
            
        return d
        
    if filter_blurry_frames:
        # We need all unique candidate indices to be present for the next step
        frameIDX_frame = ensure_dict_indices(frameIDX_frame, target_idxs_to_read)
    else:
        # We only need the base indices to be present
        frameIDX_frame = ensure_dict_indices(frameIDX_frame, base_target_idxs)

    # 4. For each window, compute blurriness and choose the sharpest
    def choose_sharpest(d, windows, total_frames):
        chosen_indices = []
        all_idx_score = {} # avoid re-scoring the same frame
        for base_idx, cand_list in zip(base_target_idxs, windows):
            idx_score = {}
            
            for idx in cand_list:
                if idx not in all_idx_score:
                    try:
                        frame = d[idx]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # NOTE: Blur metric returns LOWER value for SHARPER image
                        score = calculate_blurriness(rgb) 
                    except:
                        score = float('inf') # Highest score (worst blur) for safety
                    all_idx_score[idx] = score
                else:
                    score = all_idx_score[idx]
                
                idx_score[idx] = score
            
            scores = list(idx_score.values())
            same_score = len(set(scores)) == 1
            
            if same_score:
                # Tie-breaker logic (refactored):
                # 1. Prefer frame 0
                if 0 in cand_list:
                    best_idx = 0
                # 2. Else, prefer last frame
                elif total_frames - 1 in cand_list:
                    best_idx = total_frames - 1
                # 3. Else, select the original target frame index (center of the window)
                else:
                    best_idx = base_idx # The original index that generated the window
            else:
                # lower score is sharper in 'calculate_blurriness'. We use min().
                best_idx = min(idx_score, key=idx_score.get)
            
            chosen_indices.append(best_idx)
            
        return chosen_indices
    
    
    if filter_blurry_frames:
        chosen_indices = choose_sharpest(frameIDX_frame, frame_windows, total_frames)
    else:
        chosen_indices = base_target_idxs

    # 5. Final frame assembly
    # Re-order/select frames based on the chosen indices
    frames = [frameIDX_frame[idx] for idx in chosen_indices]
    
    # At this point, the proper frames have been selected. We only need to:
    # - resize, convert them to tensor and rotate if necessary
    if new_width != 0: # RESIZE
        frames = [cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA) for frame in frames]
    
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames] # RGB
    tensor_frames = []
    for frame in frames:
        tensor_frame = torch.from_numpy(frame.astype(np.float32) / 255.0)
        
        if required_rotation != 0:
            if required_rotation == 90:
                tensor_frame = torch.rot90(tensor_frame, k=3, dims=(0, 1)) # 90 degrees clockwise
            elif required_rotation == 180:
                tensor_frame = torch.rot90(tensor_frame, k=2, dims=(0, 1))
            elif required_rotation == 270:
                tensor_frame = torch.rot90(tensor_frame, k=1, dims=(0, 1)) # 270 clockwise
        
        tensor_frame = tensor_frame.unsqueeze(0)
        tensor_frames.append(tensor_frame)
    
    # Final sanity: ensure exact lengths; if underfilled (rare if video ended early), pad by repeating last frame
    def ensure_length(lst, target):
        if len(lst) == 0:
            raise ValueError("No frames available to form the requested output.")
        if len(lst) >= target:
            return lst[:target]
        last = lst[-1]
        while len(lst) < target:
            lst.append(last)
        return lst
        
    tensor_frames = ensure_length(tensor_frames, extract_N_frames)
    
    return torch.cat(tensor_frames, dim=0) # (N, H, W, C)
    
# ------------------------


def loadImageFromUrl(url):
    if url.startswith("data:image/"):
        i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
    elif url.startswith("s3://"):
        s3 = boto3.client('s3')
        bucket, key = url.split("s3://")[1].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        i = Image.open(io.BytesIO(obj['Body'].read()))
    else:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise Exception(response.text)

        i = Image.open(io.BytesIO(response.content))

    i = ImageOps.exif_transpose(i)

    if i.mode != "RGBA":
        i = i.convert("RGBA")

    # recreate image to fix weird RGB image
    alpha = i.split()[-1]
    image = Image.new("RGB", i.size, (0, 0, 0))
    image.paste(i, mask=alpha)

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    return (image, mask)

# --- Node class ---
class SILVER_FL_BooruBrowser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                # query options
                "site": (["Gelbooru","Danbooru","E621"], {}),
                "AND_tags": ("STRING", {"default": "frieren,", "multiline": False}),
                "OR_tags": ("STRING", {"default": "", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "limit": ("INT", {"default": 20, "min": 1, "max": 100}),
                "page": ("INT", {"default": 1, "min": 1}),
                "Safe": ("BOOLEAN", {"default": True}),
                "Questionable": ("BOOLEAN", {"default": True}),
                "Explicit": ("BOOLEAN", {"default": True}),
                "Order": (["Date","Score"], {}),
                
                # ui options
                "select_random_result": ("BOOLEAN", {"default": False}),
                "VIDEO_audio_muted": ("BOOLEAN", {"default": False}),
                "thumbnail_size": ("INT", {"default": 240, "min": 80, "max": 1024}),
                "thumbnail_quality": (["Low","Normal","High"], {}),
                
                # api options
                "gelbooru_user_id": ("STRING", {"default": ""}),
                "gelbooru_api_key": ("STRING", {"default": ""}),
                "danbooru_user_id": ("STRING", {"default": ""}),
                "danbooru_api_key": ("STRING", {"default": ""}),
                "e621_user_id": ("STRING", {"default": ""}),
                "e621_api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "selected_url": ("STRING", {"default": ""}),
                "selected_img_tags": ("STRING", {"default": ""}),
                "current_time_ms": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "file_url", "tags", "video_current_position")
    FUNCTION = "browse_booru"
    CATEGORY = "Booru"
    DESCRIPTION = """
Quickly retrieve images from Gelbooru/Danbooru/E621 without leaving ComfyUI.

Notes: 
    - Does not work with videos/animations (TIP: ensure you have 'animated' in exclude_tags)
    - 'AND_tags' , 'OR_tags' and 'exclude_tags' are single line comma separated booru tags
    - user_id for Danbooru/E621 is actually your login id
    - Searching on Danbooru is mostly useless unless you have a Gold account (you are limited to 2 tags)
    - 'select_random_result' will return a random image from the current results (ignores current selection). You should set limit to 50 or even 100 when using this feature.
"""

    def browse_booru(self, site, AND_tags, OR_tags, exclude_tags, limit, page, Safe, Questionable, Explicit, Order,
        select_random_result, VIDEO_audio_muted, thumbnail_size, thumbnail_quality,
        gelbooru_user_id, gelbooru_api_key, danbooru_user_id, danbooru_api_key, e621_user_id, e621_api_key, 
        selected_url="", selected_img_tags="", current_time_ms=0):
        
        empty_img = torch.zeros(1,1,1,3)
        file_url = selected_url if not select_random_result else RANDOMLY_SELECTED_URL
        if not file_url:
            return (empty_img, "", "", current_time_ms)
            
        try:
            is_video = isVideo(file_url)
            if not is_video:
                img, mask = loadImageFromUrl(file_url)
            else:
                if FFMPEG_available:
                    img = select_single_frame(file_url, current_time_ms if not select_random_result else 0)
                else:
                    print("[SILVER_FL_BooruBrowser] WARNING: FFMPEG not available! Please add 'animated' to exclude_tags until you install it. Output image will be blank.")
                    img = empty_img
            
            img_tags = ", ".join([tag.strip() for tag in selected_img_tags.lower().replace(',', ' ').split(' ') if tag.strip() != ''])
            
            return (img, file_url, img_tags, current_time_ms)
        except Exception as e:
            print("[SILVER_FL_BooruBrowser] error loading file_url:", e)
            return (empty_img, file_url, "", current_time_ms)

    @classmethod
    def IS_CHANGED(cls, site, AND_tags, OR_tags, exclude_tags, limit, page, Safe, Questionable, Explicit, Order,
        select_random_result, VIDEO_audio_muted, thumbnail_size, thumbnail_quality,
        gelbooru_user_id, gelbooru_api_key, danbooru_user_id, danbooru_api_key, e621_user_id, e621_api_key, 
        selected_url="", selected_img_tags="", current_time_ms=0):
        # Node is considered changed when a thumbnail selection modifies selected_url OR when select_random_result is True
        # comfyUI assumes the node is changed when the output of this function is different than the last time it ran
        # so we need to retrieve the random url from here when select_random_result and might as well turn it empty string here as well when not select_random_result
        if select_random_result:
            global RANDOMLY_SELECTED_URL
            RANDOMLY_SELECTED_URL = random.choice(CURRENT_URLS)
            return RANDOMLY_SELECTED_URL + str(current_time_ms)
        else:
            RANDOMLY_SELECTED_URL = ""
            return selected_url + str(current_time_ms)

    @classmethod
    def VALIDATE_INPUTS(cls, site, AND_tags, OR_tags, exclude_tags, limit, page, Safe, Questionable, Explicit, Order,
        select_random_result, VIDEO_audio_muted, thumbnail_size, thumbnail_quality,
        gelbooru_user_id, gelbooru_api_key, danbooru_user_id, danbooru_api_key, e621_user_id, e621_api_key, 
        selected_url="", selected_img_tags="", current_time_ms=0):
        
        s = ""
        if page < 1:
            s = "[SILVER_FL_BooruBrowser] Page must be >= 1"
        if limit < 1 or limit > 100:
            s = "[SILVER_FL_BooruBrowser] Limit must be between 1 and 100"
        if site == "Gelbooru" and (gelbooru_user_id == "" or gelbooru_api_key == ""):
            s = "[SILVER_FL_BooruBrowser] Gelbooru requires both 'gelbooru_user_id' and 'gelbooru_api_key'"
        
        if s != "":
            print(s)
            return s
        
        return True



class SILVER_Online_Video_Frame_Extractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_url": ("STRING", {"default": "", "multiline": False}),
                "extract_time_start_ms": ("INT", {"default": 0, "min": 0, "max": 500000}),
                "extract_duration_range_ms": ("INT", {"default": 5000, "min": 500, "max": 120000}),
                "extract_N_frames": ("INT", {"default": 81, "min": 1, "max": 720}),
                "frame_max_width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "frame_max_height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "filter_blurry_frames": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "get_frames"
    CATEGORY = "Booru"
    DESCRIPTION = "EXPERIMENTAL"

    def get_frames(self, file_url, extract_time_start_ms, extract_duration_range_ms, extract_N_frames, frame_max_width, frame_max_height, filter_blurry_frames):
        empty_img = torch.zeros(1,1,1,3)
        if not file_url:
            print("[SILVER_Online_Video_Frame_Extractor] invalid file_url")
            return (empty_img,)
        try:
            if isVideo(file_url):
                if FFMPEG_available:
                    selected_frames = extract_video_frames(file_url, extract_time_start_ms, extract_duration_range_ms, extract_N_frames, frame_max_width, frame_max_height, filter_blurry_frames)
                    return (selected_frames,)
                else:
                    print("[SILVER_Online_Video_Frame_Extractor] WARNING: FFMPEG not available! Output image will be blank.")
                    return (empty_img,)
            else:
                print("[SILVER_Online_Video_Frame_Extractor] file_url is not a video! Returning single image.")
                img, mask = loadImageFromUrl(file_url)
                return (img,)
        except Exception as e:
            print(f"[SILVER_Online_Video_Frame_Extractor] failed to extract frames from: {file_url}\nError: {e}")
            return (empty_img,)



        
    

# --- HTTP endpoints for the UI (search + thumb proxy) ---
@PromptServer.instance.routes.post("/silver_fl_booru/search")
async def api_booru_search(request):
    """
    Request JSON body:
    {
      "site": "Gelbooru" or "Danbooru" or "E621",
      "AND_tags": "tag1, tag2",
      "OR_tags": "tag1, tag2",
      "exclude_tags": "animated,",
      "limit": int,
      "page": int,
      "Safe": bool,
      "Questionable": bool,
      "Explicit": bool,
      "Order": "Date" or "Score",
      "thumbnail_quality": "Low" or "Normal" or "High",
      "gelbooru_user_id": "",
      "gelbooru_api_key": "",
      "danbooru_user_id": "",
      "danbooru_api_key": "",
      "e621_user_id": "",
      "e621_api_key": ""
    }
    Returns: JSON { "posts": [ {id, isVideo, file_url, preview_url, tags, width, height, source} ... ] }
    """
    global CURRENT_URLS
    try:
        data = await request.json()
        
        site = data.get("site", "Gelbooru")
        AND_tags = data.get("AND_tags", "")
        OR_tags = data.get("OR_tags", "")
        exclude_tag = data.get("exclude_tags", "")
        limit = int(data.get("limit", 20))
        page = int(data.get("page", 1))
        Safe = bool(data.get("Safe", True))
        Questionable = bool(data.get("Questionable", True))
        Explicit = bool(data.get("Explicit", True))
        Order = data.get("Order", "Date")
        thumbnail_quality = data.get("thumbnail_quality", "Low")
        
        gelbooru_user_id = data.get("gelbooru_user_id", "")
        gelbooru_api_key = data.get("gelbooru_api_key", "")
        danbooru_user_id = data.get("danbooru_user_id", "")
        danbooru_api_key = data.get("danbooru_api_key", "")
        e621_user_id = data.get("e621_user_id", "")
        e621_api_key = data.get("e621_api_key", "")
        
        # fix page by site if necessary
        if site == "Gelbooru":
            page -= 1 # Gelbooru API expects the fist page to be 0 but this node's minimum page value is 1
        
        # Normalize tags
        def normalize_tags(tag_str):
            tags = tag_str.rstrip(',').rstrip(' ')
            tags = tags.split(',')
            tags = [item.strip().replace(' ', '_').replace('\\', '') for item in tags]
            return [item for item in tags if item]
        
        
        AND_tags = normalize_tags(AND_tags)
        OR_tags = normalize_tags(OR_tags)
        exclude_tags = normalize_tags(exclude_tag)
        
        # Build tag query depending on site
        tag_query = []
        if site == "Gelbooru":
            if AND_tags:
                tag_query.append('+'.join(AND_tags))
            if OR_tags:
                if len(OR_tags) > 1:
                    tag_query.append('{' + ' ~ '.join(OR_tags) + '}')
                else:
                    tag_query.append(OR_tags[0])
            if exclude_tags:
                tag_query.append('+'.join('-' + t for t in exclude_tags))
        elif site == "E621":
            tag_query.extend(AND_tags)
            if OR_tags:
                # Each OR tag must be prefixed with "~"
                tag_query.extend([f"~{t}" for t in OR_tags])
            if exclude_tags:
                tag_query.extend(["-" + t for t in exclude_tags])
        elif site == "Danbooru":
            # Danbooru uses space-separated tags, with `-` prefix for exclusion
            tag_query.extend(AND_tags)
            if OR_tags:
                # Each OR tag must be prefixed with "~"
                tag_query.extend([f"~{t}" for t in OR_tags])
            if exclude_tags:
                tag_query.extend(["-" + t for t in exclude_tags])
        
        
        # Rating filtering
        if site == "Gelbooru":
            if not Safe:
                tag_query.append("-rating:general")
            if not Questionable:
                tag_query.append("-rating:questionable -rating:sensitive")
            if not Explicit:
                tag_query.append("-rating:explicit")
        elif site == "E621":
            rating_filters = []
            if not Safe:
                rating_filters.append("-rating:s")
            if not Questionable:
                rating_filters.append("-rating:q")
            if not Explicit:
                rating_filters.append("-rating:e")
            tag_query.extend(rating_filters)
        elif site == "Danbooru":
            if not Safe:
                tag_query.append("-rating:s")
            if not Questionable:
                tag_query.append("-rating:q")
            if not Explicit:
                tag_query.append("-rating:e")
        
        # Apply ordering
        if Order == "Score":
            if site == "Gelbooru":
                tag_query.append("sort:score:desc")
            elif site == "Danbooru":
                tag_query.append("order:score_desc")
            elif site == "E621":
                tag_query.append("order:score_desc")
        
        # Base URL
        if site == "Gelbooru":
            base_url = "https://gelbooru.com/index.php"
        elif site == "E621":
            base_url = "https://e621.net/posts.json"
        elif site == "Danbooru":
            base_url = "https://danbooru.donmai.us/posts.json"
        
        user_id = (
            gelbooru_user_id if site == "Gelbooru" else
            danbooru_user_id if site == "Danbooru" else
            e621_user_id
        )
        api_key = (
            gelbooru_api_key if site == "Gelbooru" else
            danbooru_api_key if site == "Danbooru" else
            e621_api_key
        )
        
        
        # Build request
        params = {}
        headers = {"User-Agent": USER_AGENT}
        url = base_url
        
        if site == "Gelbooru":
            query = (
                f"page=dapi&s=post&q=index&tags="
                f"{'+'.join(tag_query)}"
                f"&limit={limit}&pid={page}&json=1&api_key={api_key}&user_id={user_id}"
            )
            url = f"{base_url}?{query}"
            url = re.sub(r"\++", "+", url)
        else:
            # E621 and Danbooru both use JSON endpoints and accept params
            params["limit"] = limit
            params["page"] = page
            params["tags"] = " ".join(tag_query)
            if user_id and api_key:
                params["login"] = user_id
                params["api_key"] = api_key
        
        resp = requests.get(url, params=(params if site != "Gelbooru" else None), headers=headers, timeout=10)
        resp.raise_for_status()
        
        posts = []
        if site == "Gelbooru":
            posts = resp.json().get("post", [])
        elif site == "E621":
            posts = resp.json().get("posts", [])
        elif site == "Danbooru":
            posts = resp.json()  # Danbooru returns an array of post objects   
        
        #print(f"[SILVER_FL_BooruBrowser] url: {url}")
        #print(f"[SILVER_FL_BooruBrowser] params: {params}")
        #print(f"[SILVER_FL_BooruBrowser] posts: {posts}")
        
        CURRENT_URLS.clear()
        
        out_posts = []
        for post in posts:
            id = "" # optional
            file_url = "" # required
            preview_url = "" # required
            tags = "" # required - must be a string with tags separated by spaces
            source = "" # optional
            
            if site == "Gelbooru":
                id = post.get("id", "")
                source = post.get("source", "")
                file_url = post.get("file_url", "") or source # when file_url is not available use source instead
                preview_url = ( (post.get("sample_url", "") or post.get("preview_url", "")) if thumbnail_quality == "High" else (post.get("preview_url", "") or post.get("sample_url", "")) ) or file_url # as a last resort: set it to file_url
                tags = post.get("tags", "")
            
            elif site == "E621":
                id = str(post.get("id", ""))
                source = post.get("sources", [None])[0] if post.get("sources") else ""
                file_url = post.get("file", {}).get("url", "") or source
                preview_url = ( (post.get("sample", {}).get("url", "") or post.get("preview", {}).get("url", "")) if thumbnail_quality == "High" else (post.get("preview", {}).get("url", "") or post.get("sample", {}).get("url", "")) ) or file_url
                # Tags in E621 are grouped by category (general, species, etc.)
                tag_groups = post.get("tags", {})
                all_tags = []
                for group_tags in tag_groups.values():
                    all_tags.extend(group_tags)
                tags = " ".join(all_tags)
            
            elif site == "Danbooru":
                id = str(post.get("id", ""))
                source = post.get("source", "")
                # Danbooru returns fields like file_url, large_file_url, preview_file_url
                file_url = post.get("large_file_url", "") or post.get("file_url", "") or source
                preview_url = post.get("preview_file_url", "") # non-Gold users won't get a 'preview_file_url' for gold-exclusive files - so we take advantage of that and don't fallback to file_url here
                tags = post.get("tag_string", "")  # Danbooru’s tag string                
                
                # --- Danbooru preview_url selection with thumbnail_quality support ---
                if preview_url and thumbnail_quality != "Low":
                    # safe extraction of danbooru variants (handles missing / malformed media_asset)
                    media_asset = post.get("media_asset")
                    if isinstance(media_asset, dict):
                        variants = media_asset.get("variants") if isinstance(media_asset.get("variants"), list) else []
                    else:
                        variants = []
                    
                    variant_map = {}
                    for v in variants:
                        if isinstance(v, dict):
                            t = v.get("type")
                            u = v.get("url")
                            if t and u:
                                variant_map[t] = u
                    
                    preview_url = ( (variant_map.get("720x720") or variant_map.get("360x360")) if thumbnail_quality == "High" else variant_map.get("360x360") ) or preview_url
            
            
            #print(f"[SILVER_FL_BooruBrowser] preview_url: {preview_url}")
            #print(f"[SILVER_FL_BooruBrowser] file_url: {file_url}")
            
            if file_url and preview_url and tags:
                CURRENT_URLS.append(file_url)
                out_posts.append({
                    "id": id,
                    "isVideo": isVideo(file_url),
                    "file_url": file_url,
                    "preview_url": preview_url,
                    "tags": tags,
                    "source": source,
                    "site": site,
                    "user_id": user_id,
                    "api_key": api_key
                })
        
        
        return web.json_response({"posts": out_posts})
    except Exception as e:
        print("[SILVER_FL_BooruBrowser] /search error:", e)
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/silver_fl_booru/thumb")
async def api_booru_thumb(request):
    """
    Request JSON body: { "url": "<image url>", "size": int, "thumbnail_quality": string, "site": string, "user_id": string, "api_key": string }
    Returns JPEG thumbnail bytes (image/jpeg) OR PNG thumbnail bytes (image/png)
    """
    try:
        data = await request.json()
        
        url = data.get("url", "")
        size = int(data.get("size", 240))
        thumbnail_quality = data.get("thumbnail_quality", "Low")
        
        # in case its necessary to parse the url or add cookies with these
        site = data.get("site", "")    
        user_id = data.get("user_id", "")
        api_key = data.get("api_key", "")
        
        if not url:
            return web.json_response({"error": "No url provided"}, status=400)
        
        # Fetch image (timeout small)
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, timeout=8, stream=True, headers=headers)
        resp.raise_for_status()
        
        #img = Image.open(io.BytesIO(resp.content)) # this is slower
        img = Image.open(resp.raw)
        
        format = "PNG" if thumbnail_quality == "High" else "JPEG"
        params = {}
        
        if format == "JPEG":
            params["quality"] = 95 if thumbnail_quality == "Normal" else 85
        
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format=format, **params)
        buf.seek(0)
        
        return web.Response(body=buf.read(), content_type = 'image/png' if format == "PNG" else 'image/jpeg')
    except Exception as e:
        print("[SILVER_FL_BooruBrowser] /thumb error:", e)
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/silver_fl_booru/videoprobe")
async def api_video_probe(request):
    """
    Right now we only need duration and rotation so that's all we are probing for here.
    Returns: JSON { "duration_ms": INT, "rotation": INT }
    """
    try:
        data = await request.json()
        file_url = data.get("file_url", "")
        duration_ms, rotation = get_video_duration_and_rotation(file_url)
        return web.json_response({"duration_ms": duration_ms, "rotation": rotation})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/silver_fl_booru/videostream")
async def api_video_stream(request):
    try:
        file_url = request.query.get("file_url")
        if file_url is None:
            return web.json_response({"error": "file_url required"}, status=400)
        
        # currently not used cause its handled via browser's native <video>
        seek_time_str = request.query.get("seek_time")
        seek_time = 0.0
        if seek_time_str:
            try:
                seek_time = float(seek_time_str)
            except ValueError:
                pass
        
        range_header = request.headers.get("Range") # Forward range header if present
        headers = {}
        if range_header:
            headers["Range"] = range_header
        
        try:
            session = aiohttp.ClientSession()
            resp = await session.get(file_url, headers=headers, ssl=file_url.startswith("https"))
            status = resp.status
            
            response_headers = {
                "Content-Type": resp.headers.get("Content-Type", "application/octet-stream"),
            }
            # Pass through range-related headers
            for h in ["Content-Range", "Accept-Ranges", "Content-Length"]:
                if h in resp.headers:
                    response_headers[h] = resp.headers[h]
            
            # Create a streaming response
            stream_resp = web.StreamResponse(
                status=status,
                headers=response_headers
            )
            await stream_resp.prepare(request)
            
            # Stream the data
            async for chunk in resp.content.iter_chunked(1024 * 64):
                try:
                    await stream_resp.write(chunk)
                except ConnectionResetError:
                    break
                except Exception as e2:
                    #print(f"[SILVER_FL_BooruBrowser] Write error: {e2}")
                    break
            
            await stream_resp.write_eof()
            return stream_resp
        finally:
            # Ensure the aiohttp response is released and the session is closed
            if 'resp' in locals() and resp:
                resp.release()
            await session.close()
    
    except Exception as e:
        if 'session' in locals() and session and not session.closed:
             await session.close()
        return web.json_response({"error": str(e)}, status=500)


NODE_CLASS_MAPPINGS = {
    "SILVER_FL_BooruBrowser": SILVER_FL_BooruBrowser,
    "SILVER_Online_Video_Frame_Extractor": SILVER_Online_Video_Frame_Extractor,
}

# Provide a display name mapping so it looks nice in the node list (optional)
NODE_DISPLAY_NAME_MAPPINGS = {
    "SILVER_FL_BooruBrowser": "[Silver] Booru Browser",
    "SILVER_Online_Video_Frame_Extractor": "[Silver] Online Video Frame Extractor (REQUIRES FFMPEG)",
}


