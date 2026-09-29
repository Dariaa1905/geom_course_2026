import numpy as np

def pad_image(image: np.ndarray, pad: int, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.pad(
            image,
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0,
        )

    if mode == "edge":
        return np.pad(
            image,
            ((pad, pad), (pad, pad)),
            mode="edge",
        )

    raise ValueError(f"Unknown padding mode: {mode}")


def box_blur(
    image: np.ndarray,
    kernel_size: int,
    mode: str,
) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    pad = kernel_size // 2
    padded = pad_image(image, pad, mode)

    result = np.empty(image.shape, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[
                i:i + kernel_size,
                j:j + kernel_size,
            ]

            result[i, j] = np.mean(window)

    return result


def make_test_image(size: int = 40) -> np.ndarray:
    import random
    random.seed(0)
    img = np.zeros((size, size))
    for i in range(size):
        row = np.zeros(size)
        for j in range(size):
            base = 220.0 if (i // 5 + j // 5) % 2 == 0 else 40.0
            noise = random.uniform(-15, 15)
            row[j] =  max(0.0, min(255.0, base + noise))
        img[i] =  row
    return img

def main():
    import numpy as np
    import matplotlib.pyplot as plt
 
    image = make_test_image(size=40)
 
    blurred_k3 = box_blur(image, kernel_size=3, mode="edge")
    blurred_k9 = box_blur(image, kernel_size=9, mode="zero")
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["исходное", "Blur k=3 (edge)", "Blur k=9 (zero)"]
    data = [image, blurred_k3, blurred_k9]
 
    for ax, title, d in zip(axes, titles, data):
        ax.imshow(d, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
 
    plt.show()




