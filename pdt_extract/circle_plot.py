import numpy as np
import matplotlib.pyplot as plt  # Only for displaying the image (optional)

def draw_circle_on_ndarray(image, center, radius, color=255):
    """
    Draw a circle on a 2D ndarray (grayscale image).

    Args:
        image (np.ndarray): Grayscale image represented as a NumPy ndarray.
        center (tuple): Center coordinates (x, y) of the circle.
        radius (int): Radius of the circle in pixels.
        color (int): Pixel intensity (default is white, 255).

    Returns:
        np.ndarray: Image with the drawn circle.
    """
    # Create a copy of the input image to avoid modifying the original
    image_with_circle = image.copy()

    # Generate a grid of x and y coordinates
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]

    # Create a mask for the circle using the equation of a circle
    mask = ((x - center[0])**2 + (y - center[1])**2 <= radius**2)

    # Draw the circle on the image by setting the corresponding pixels to the specified color
    image_with_circle[mask] = color

    return image_with_circle


def place_circle():
    # Example usage:
    # Create a sample grayscale image (you should load your own image)
    image = np.zeros((200, 200), dtype=np.uint8)

    center_coordinates = (100, 100)   # Example center coordinates (x, y)
    circle_radius = 0                # Example radius in pixels

    result_image = draw_circle_on_ndarray(image, center_coordinates, circle_radius)

    # Display the result (optional)
    plt.imshow(result_image, cmap='gray')
    plt.axis('off')
    plt.show()