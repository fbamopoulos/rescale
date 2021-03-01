from os import path, walk
import cv2
import numpy as np
from pathlib import Path

TARGET_DIR = './rescaled_images'
SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png']
THRESHOLD = 2560
INTERPOLATION = cv2.INTER_AREA
SUFFIX = ''


def imread_custom(image_file_path, *args, **kwargs):
    numpy_array = np.fromfile(image_file_path, dtype=np.uint8)
    image_bgr = cv2.imdecode(numpy_array,  *args, **kwargs)
    return image_bgr


def imwrite_custom(target_path, image_bgr):
    extension = target_path.rsplit('.', maxsplit=1)[-1]
    is_success, im_buf_arr = cv2.imencode('.'+extension, image_bgr)
    im_buf_arr.tofile(target_path)


def rescale_images(target_dir):
    _, _, filenames = next(walk(target_dir))
    for filename in filenames:
        filename_no_extension, file_extension = filename.rsplit('.', maxsplit=1)
        if file_extension.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # standard solution
        # input_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        # unicode solution
        file_path = path.join(target_dir, filename)
        input_image = imread_custom(file_path, cv2.IMREAD_UNCHANGED)
        if input_image is None:
            print('Could not read {}'.format(file_path))
            continue
        input_height = input_image.shape[0]
        input_width = input_image.shape[1]
        print('Path:', file_path)
        print('Resolution:', input_height, 'x', input_width)
        if not (input_height > THRESHOLD or input_width > THRESHOLD):
            print('Image dimensions are smaller than the threshold')
            print()
            continue
        biggest_dimension = max([input_height, input_width])
        scale_factor = THRESHOLD / biggest_dimension
        new_height = int(input_height*scale_factor)
        new_width = int(input_width*scale_factor)
        print('Scaled resolution:', new_height, 'x', new_width)
        new_resolution = (new_height, new_width)
        # for some reason cv2.resize expects the target resolution in (width, height) so I reverse it
        scaled_image = cv2.resize(input_image, new_resolution[::-1], interpolation=INTERPOLATION)
        new_file_name = '{}{}.{}'.format(filename_no_extension, SUFFIX, file_extension)
        new_file_path = path.join(target_dir, new_file_name)
        # standard solution
        # cv2.imwrite(new_file_path, scaled_image)
        # unicode solution
        imwrite_custom(new_file_path, scaled_image)
        print('Written to {}'.format(new_file_path))
        print()


if __name__ == '__main__':
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    rescale_images(TARGET_DIR)
