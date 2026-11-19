import io

from PIL import Image

from cp2.official_source_eval import exclude_duplicate_images


def encoded(color, compression):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), color).save(buffer, format="PNG", compress_level=compression)
    return buffer.getvalue()


def test_dedup_uses_rgb_pixels_instead_of_compressed_bytes():
    first = encoded((12, 24, 60), 0)
    same = encoded((12, 24, 60), 9)
    different = encoded((12, 24, 61), 9)
    assert first != same
    images = {"fit": first, "match": same, "heldout": different}
    assert exclude_duplicate_images(images, {"fit"}, {"match", "heldout"}) == {"match"}
