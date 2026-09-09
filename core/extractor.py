import io
import os
import exifread
import numpy as np
from PIL import Image

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".dng", ".raf", ".rw2", ".orf", ".pef"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

def safe_float(val):
    """Safely converts exifread Ratio, list, or string to float."""
    if val is None:
        return None
    val_str = str(val).strip("[] ")
    try:
        if "/" in val_str:
            num, den = val_str.split("/")
            den_f = float(den)
            return round(float(num) / den_f, 4) if den_f != 0 else None
        return round(float(val_str), 4)
    except Exception:
        return None

def format_shutter_speed(sec):
    """Formats float seconds into standard photography shutter notation (e.g. 1/250s)."""
    if sec is None or sec <= 0:
        return "N/A"
    if sec >= 1.0:
        return f"{sec:.1f}s"
    reciprocal = round(1.0 / sec)
    return f"1/{reciprocal}s"

def extract_preview_histogram(file_source, filename):
    """
    Extracts embedded preview thumbnail to compute tonal curves,
    shadow clipping (crushed blacks), and highlight blowout without demosaicing.
    """
    ext = os.path.splitext(filename)[1].lower()
    img = None

    try:
        if ext in RAW_EXTENSIONS and HAS_RAWPY:
            # Fast extraction of embedded camera preview via rawpy
            if isinstance(file_source, (str, os.PathLike)):
                with rawpy.imread(file_source) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(thumb.data))
                    except Exception:
                        pass
            elif hasattr(file_source, "seek"):
                file_source.seek(0)
                with rawpy.imread(file_source) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(thumb.data))
                    except Exception:
                        pass
                file_source.seek(0)
        
        # Fallback for standard images or if rawpy preview failed
        if img is None:
            if isinstance(file_source, (str, os.PathLike)):
                img = Image.open(file_source)
            else:
                file_source.seek(0)
                img = Image.open(file_source)

        img = img.convert("L")  # Convert to Grayscale Luminance
        img.thumbnail((400, 400))  # Downsample for ultra-fast calculation
        arr = np.array(img, dtype=np.uint8)
        
        total_pixels = arr.size
        shadow_clipped_pct = float(np.sum(arr <= 2) / total_pixels * 100.0)
        highlight_clipped_pct = float(np.sum(arr >= 253) / total_pixels * 100.0)
        mean_luminance = float(np.mean(arr))
        rms_contrast = float(np.std(arr))
        
        # 256-bin histogram
        hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
        
        return {
            "shadow_clipped_pct": round(shadow_clipped_pct, 2),
            "highlight_clipped_pct": round(highlight_clipped_pct, 2),
            "mean_luminance": round(mean_luminance, 1),
            "rms_contrast": round(rms_contrast, 1),
            "luminance_hist": hist.tolist()
        }
    except Exception:
        return {
            "shadow_clipped_pct": 0.0,
            "highlight_clipped_pct": 0.0,
            "mean_luminance": 128.0,
            "rms_contrast": 50.0,
            "luminance_hist": None
        }

def process_single_image(file_source, filename):
    """Parses EXIF metadata and preview metrics for a single image."""
    data = {
        "filename": os.path.basename(filename),
        "extension": os.path.splitext(filename)[1].lower(),
        "aperture": None,
        "shutter_speed": None,
        "shutter_str": "N/A",
        "iso": None,
        "focal_length": None,
        "camera_model": "Unknown",
        "lens_model": "Unknown",
        "exposure_bias": 0.0,
        "date_taken": None,
        "hour_of_day": None,
        "filepath": str(file_source) if isinstance(file_source, (str, os.PathLike)) else None,
    }

    # Extract EXIF tags
    try:
        if isinstance(file_source, (str, os.PathLike)):
            with open(file_source, "rb") as f:
                tags = exifread.process_file(f, details=False)
        else:
            file_source.seek(0)
            tags = exifread.process_file(file_source, details=False)
            file_source.seek(0)
            
        data["aperture"] = safe_float(tags.get("EXIF FNumber"))
        data["shutter_speed"] = safe_float(tags.get("EXIF ExposureTime"))
        data["shutter_str"] = format_shutter_speed(data["shutter_speed"])
        data["iso"] = safe_float(tags.get("EXIF ISOSpeedRatings"))
        data["focal_length"] = safe_float(tags.get("EXIF FocalLength"))
        data["exposure_bias"] = safe_float(tags.get("EXIF ExposureBiasValue")) or 0.0
        data["exposure_program"] = str(tags.get("EXIF ExposureProgram", "Unknown")).strip()
        data["exposure_mode"] = str(tags.get("EXIF ExposureMode", "Unknown")).strip()
        data["metering_mode"] = str(tags.get("EXIF MeteringMode", "Unknown")).strip()
        data["white_balance"] = str(tags.get("EXIF WhiteBalance", "Unknown")).strip()
        
        fl_35 = safe_float(tags.get("EXIF FocalLengthIn35mmFilm"))
        if fl_35:
            data["focal_length_35mm"] = fl_35
        elif data["focal_length"]:
            data["focal_length_35mm"] = round(data["focal_length"] * 1.5, 1)
        else:
            data["focal_length_35mm"] = None
            
        flash_str = str(tags.get("EXIF Flash", "")).strip()
        data["flash_fired"] = bool("fired" in flash_str.lower() and "not fire" not in flash_str.lower())
        data["flash_status"] = "Fired" if data["flash_fired"] else "Off"
        
        if "Image Model" in tags:
            data["camera_model"] = str(tags["Image Model"]).strip()
        if "EXIF LensModel" in tags:
            data["lens_model"] = str(tags["EXIF LensModel"]).strip()
        if "EXIF DateTimeOriginal" in tags:
            dt_str = str(tags["EXIF DateTimeOriginal"]).strip()
            data["date_taken"] = dt_str
            try:
                # Format: "YYYY:MM:DD HH:MM:SS"
                time_part = dt_str.split(" ")[1]
                data["hour_of_day"] = int(time_part.split(":")[0])
            except Exception:
                pass
    except Exception as e:
        data["error"] = str(e)

    # Extract curve & clipping metrics
    curve_metrics = extract_preview_histogram(file_source, filename)
    data.update(curve_metrics)

    return data

def get_thumbnail_image(file_source_or_path, max_size=(320, 320)):
    """
    Returns a small PIL Image thumbnail for gallery display.
    Very fast: extracts embedded JPEG thumbnail from RAW files.
    """
    try:
        if isinstance(file_source_or_path, (str, os.PathLike)):
            ext = os.path.splitext(file_source_or_path)[1].lower()
            if ext in RAW_EXTENSIONS and HAS_RAWPY:
                with rawpy.imread(file_source_or_path) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(thumb.data))
                            img.thumbnail(max_size)
                            return img
                    except Exception:
                        pass
            img = Image.open(file_source_or_path)
            img.thumbnail(max_size)
            return img
        elif hasattr(file_source_or_path, "seek"):
            file_source_or_path.seek(0)
            ext = os.path.splitext(getattr(file_source_or_path, "name", ""))[1].lower()
            if ext in RAW_EXTENSIONS and HAS_RAWPY:
                with rawpy.imread(file_source_or_path) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(thumb.data))
                            img.thumbnail(max_size)
                            file_source_or_path.seek(0)
                            return img
                    except Exception:
                        pass
                file_source_or_path.seek(0)
            img = Image.open(file_source_or_path)
            img.thumbnail(max_size)
            file_source_or_path.seek(0)
            return img
    except Exception:
        return None

