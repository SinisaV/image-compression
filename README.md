# Image Compression and Decompression

This Python script allows for the compression and decompression of BMP images using predictive coding. The compression method is based on the concept of prediction errors, and the decompression method reconstructs the image from the compressed data.

## Usage

1. Ensure you have Python installed on your system.
2. Install required packages by running:
```
pip install Pillow
```

4. Run the script:
```
python your_script_name.py
```

## Quick Overview

**Functions:**
- The script includes functions for reading BMP images, compressing, decompressing, and saving images.

**Sample Images:**
- Sample BMP images are provided in the "slike BMP" directory.

**Notes:**
- Ensure the image file path is correct.
- The script outputs information about compression, decompression, and image sizes.

## Example

```python
image_path = "path/to/your/image.bmp"
slika, pixel_values, Y, X, original_mode = read_bmp_image(image_path)
predicted_values, N, C, B, Bic = compress(slika, X, Y)
decImg = decompress(Bic)
dec_img_path = "decompressed.bmp"
save_as_bmp(decImg, dec_img_path, original_mode)
```
