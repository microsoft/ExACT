from typing import Optional
from PIL import Image
from io import BytesIO
import base64


class SoftBudgetTracker:
    def __init__(self, budget: float | int):
        self.budget = budget
        self.spent = 0
        return

    def spend(self, amount: float | int):
        self.spent += amount
        return

    def get_remaining(self) -> float | int:
        return self.budget - self.spent

    def get_spent(self) -> float | int:
        return self.spent


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def _pil_image_to_str(images: Optional[list[Image.Image]]) -> str:
    # used to hash the images and save to self._retrieval_cache (together with intent)
    if images is not None:
        encoded_image_str = ""
        for img in images:
            encoded_image_str += pil_to_b64(img)
    else:
        encoded_image_str = ""
    return encoded_image_str