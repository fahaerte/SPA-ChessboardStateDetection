import os
import requests
from io import BytesIO
import imutils
from decouple import config
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_and_convert_image(url):
    filename = config('IMAGE_FILENAME')
    folder_name = config('IMAGE_FOLDER')
    path = os.path.join(os.getcwd(), folder_name, filename)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(path)
    image = cv2.imread(path)
    return image


def rotate_image(image, angle):
    image = imutils.rotate(image, angle=angle)
    return image


def crop_image(image, x, y, h, w):
    image = image[y:y + h, x:x + w]
    return image


def show_tile(tile, colors):
    tile = tile.flatten()
    img = colors[tile]
    img = np.reshape(img, (20, 20, 3))
    plt.imshow(img)
    plt.show()


def show_image_and_colors(labels, centers, k):
    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]

    # reshape back to the original image dimension
    segmented_image = np.reshape(segmented_image, (232, 232, 3))

    labels_rgb_values = []
    # Show clusters
    cluster_images = []
    for i in range(0, k):
        cluster = np.full(labels.shape, i)
        im = centers[cluster]
        im = np.reshape(im, (232, 232, 3))
        labels_rgb_values.append(im[0][0])
        cluster_images.append(im)

    # Show cluster colors
    counter = 0
    _, axs = plt.subplots(1, k + 1, figsize=(15, 15))
    axs = axs.flatten()
    for img, ax in zip(cluster_images, axs):
        ax.imshow(img)
        counter += 1

    # show the image
    plt.imshow(segmented_image)
    plt.show()


def show_image(labels, centers):
    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]
    # reshape back to the original image dimension
    segmented_image = np.reshape(segmented_image, (232, 232, 3))
    plt.imshow(segmented_image)
    plt.show()


def add_color_frame(image, frame_width=8):
    image = cv2.copyMakeBorder(image, frame_width, frame_width, frame_width, frame_width, cv2.BORDER_CONSTANT, None,
                               value=(255, 255, 255))
    image = cv2.copyMakeBorder(image, frame_width, frame_width, frame_width, frame_width, cv2.BORDER_CONSTANT, None,
                               value=(0, 0, 0))
    image = cv2.copyMakeBorder(image, frame_width, frame_width, frame_width, frame_width, cv2.BORDER_CONSTANT, None,
                               value=(0, 0, 255))
    image = cv2.copyMakeBorder(image, frame_width, frame_width, frame_width, frame_width, cv2.BORDER_CONSTANT, None,
                               value=(255, 0, 0))
    return image
