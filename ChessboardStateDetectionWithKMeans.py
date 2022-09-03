from ChessboardStateDetectionUtil import *


def identify_order_of_colors(centers):

    for i in range(0, len(centers)):
        if centers[i][0] > 100 and centers[i][1] > 100 and centers[i][2] > 100:
            index_white = i
        elif centers[i][0] <= 100 and centers[i][1] <= 100 and centers[i][2] <= 100:
            index_black = i
        elif centers[i][0] > 100 and centers[i][1] <= 100 and centers[i][2] <= 100:
            index_red = i
        elif centers[i][0] <= 100 and centers[i][1] <= 100 and centers[i][2] > 100:
            index_blue = i
        else:
            print("One color couldnt be identified!")

    color_value_pairs = {
        "black": index_black,
        "blue": index_blue,
        "red": index_red,
        "white": index_white,
    }
    return color_value_pairs


def get_amout_of_color_pixels_in_tile(tile, wh_t, color_value_pairs):
    amount_blue = count_color_pixels(tile, wh_t, color_value_pairs['blue'])
    amount_red = count_color_pixels(tile, wh_t, color_value_pairs['red'])
    amount_white = count_color_pixels(tile, wh_t, color_value_pairs['white'])
    amount_black = count_color_pixels(tile, wh_t, color_value_pairs['black'])

    # how many pixels of respective color are contained in tile
    color_pixel_counts = [amount_white, amount_black, amount_blue,
                          amount_red]
    return color_pixel_counts


def count_color_pixels(tile, wh_t, color_value):
    if color_value == 0:
        return pow(wh_t, 2) - np.count_nonzero(tile)
    else:
        return np.count_nonzero(tile == color_value)


def decide_color_of_tile(color_pixel_counts): # order:  white, black, blue, red
    if color_pixel_counts[3] >= 5:
        return 'r'
    elif color_pixel_counts[2] > 50:
        return 'b'
    else:
        return 'e'


def calculate_chessboard_state(url='https://lab.bpm.in.tum.de/img/low'):
    frame_width = 8
    frame_size = 4 * frame_width
    size_with_frame = 232 + 2 * frame_size
    tile_size = 232 // 8

    image = load_and_convert_image(url)
    image = crop_image(image, y=2, x=100, h=232, w=232)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = add_color_frame(image, frame_width=frame_width)
    image_pixels = image.reshape((-1, 3))  # reshape the image to a 2D array of pixels and 3 color values (RGB)
    image_pixels = np.float32(image_pixels)  # convert to float

    # Apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    _, labels, centers = cv2.kmeans(image_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)
    labels_reshaped = np.reshape(labels, (size_with_frame, size_with_frame))
    labels_reshaped = labels_reshaped[frame_size: size_with_frame-frame_size, frame_size: size_with_frame-frame_size]
    result = []
    color_value_pairs = identify_order_of_colors(centers)
    print("(i, j) | x:x,  y:y  |  white, black, blue, red")
    print("----------------------------------------")
    for i in range(0, 8):
        for j in range(0, 8):
            # Ignore the surrounding 9 pixels: Cut 20x20 tiles
            x_start = i * tile_size + 5
            x_end = (i + 1) * tile_size - 4
            y_start = j * tile_size + 5
            y_end = (j + 1) * tile_size - 4
            tile_width = x_end - x_start
            tile = labels_reshaped[x_start: x_end, y_start: y_end]
            label_counts = get_amout_of_color_pixels_in_tile(tile, tile_width, color_value_pairs) # order:  white, black, blue, red
            color_in_tile = decide_color_of_tile(label_counts)

            print("(" + str(i) + ", " + str(j) + ") | " + str(x_start) + ":" + str(x_end) + ", " + str(
                y_start) + ":" + str(y_end) + " | " + '  '.join(str(e) for e in label_counts) + " | " + color_in_tile)

            result.append(color_in_tile)
            # show_tile(tile, centers)

    # show_image(labels_reshaped, centers)
    result = np.reshape(result, (8, 8))
    return {'chessboardstate': result.tolist()}
