import os
import shutil
import time
import imgaug.augmenters as iaa


def aug_name(dataPath: str):
    """
    Creates 33 augmented images for each image from the directory, passed by the dir_name parameter and saves them in
    the same directory under the name augmented_{name_of_image}_{unic_index}.
    :param: dir_name [str]
        Name of directory.
    :param: path [str], optional
        Full path to dir, passed by dir_name parameter, is needed if directory,
        you want work with is not in directory faceRec
        by default os.path.dirname(os.path.abspath(__file__)).
    """
    image_paths = [os.path.join(dataPath, f) for f in os.listdir(dataPath)]
    augmentations = [
        iaa.PerspectiveTransform(scale=0.15, random_state=False),
        iaa.ScaleX(1.35),
        iaa.ScaleX(0.8),
        iaa.ScaleY(1.35),
        iaa.ScaleY(0.8),
        iaa.Affine(rotate=45),
        iaa.Affine(rotate=-45),
        iaa.Fliplr(1),
        iaa.Flipud(1),
        iaa.GaussianBlur(2.5)
    ]

    add_bright = [
        iaa.Multiply(0.5),
        iaa.Multiply(1.5)
    ]

    for path in image_paths:
        image = cv2.imread(path)
        
        if image is None:
            print("Failed to read image:", path)
            continue

        if image.size == 0:
            print("Empty image:", path)
            continue

        augmented_images = []
        for i, augmentation in enumerate(augmentations):
            augmented_image = np.copy(image)
            augmented_image = augmentation(image=augmented_image)
            augmented_images.append(augmented_image)
        augmented_images.append(image)
        res_augmented_images = augmented_images.copy()
        for img in augmented_images:
            for aug in add_bright:
                aug_img = np.copy(img)
                aug_img = aug(image=aug_img)
                res_augmented_images.append(aug_img)

        for i, img in enumerate(res_augmented_images):
            img_name = os.path.join(dataPath,
                                    ''.join(["augmented_", os.path.basename(path)[:4], '_', str(i + 1), ".jpg"]))
            cv2.imwrite(img_name, img)


from PIL import Image
import cv2

def from_first_digit(string: str):
    for i in range(len(string)):
        if string[i].isdigit():
            return string[i:]
    return ''


def preprocess_filename(string: str):
    string = string[:string.rfind('_')]
    string = from_first_digit(string)
    for i in range(2):
        if string[-2:] in ['-1', '-2', '_1', '_2']:
            string = string[:-2]
    string = string.replace('-', '').replace('_', '')
    return string

# for filename in os.listdir(r'C:\Users\32233\PycharmProjects\temp\boxeswithoutall\train'):
#     processed_filename = preprocess_filename(filename)
#     if len(processed_filename) != 10:
#         print(filename)
# for filename in os.listdir(r'C:\Users\32233\PycharmProjects\temp\boxeswithoutall\valid'):
#     processed_filename = preprocess_filename(filename)
#     shutil.copyfile(rf'C:\Users\32233\PycharmProjects\temp\boxeswithoutall\valid\{filename}',
#                     rf'C:\Users\32233\PycharmProjects\temp\boxes_with_fixed_labels\val\{processed_filename}.jpg')
dir_path = r'C:\Users\32233\PycharmProjects\temp\boxes_with_fixed_labels\val'
for filename in os.listdir(dir_path):
    img = cv2.imread(os.path.join(dir_path, filename))
    img2 = cv2.resize(img,(250, 50))
    cv2.imwrite(os.path.join(r"C:\Users\32233\PycharmProjects\temp\boxes_with_fixed_labels2\val", filename), img2)