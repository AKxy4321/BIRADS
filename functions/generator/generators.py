from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def resize_and_pad_image(image, target_size=(224, 224)):
    img_shape = tf.shape(image)
    height, width = img_shape[0], img_shape[1]

    # Calculate aspect ratios
    aspect_ratio = width / height
    target_aspect_ratio = target_size[1] / target_size[0]

    # Resize while maintaining aspect ratio
    if aspect_ratio > target_aspect_ratio:
        new_width = target_size[1]
        new_height = tf.cast(tf.round(new_width / aspect_ratio), dtype=tf.int32)
    else:
        new_height = target_size[0]
        new_width = tf.cast(tf.round(new_height * aspect_ratio), dtype=tf.int32)

    resized_image = tf.image.resize(image, [new_height, new_width])

    # Add padding to match target size
    pad_height = target_size[0] - tf.shape(resized_image)[0]
    pad_width = target_size[1] - tf.shape(resized_image)[1]
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    padded_image = tf.pad(resized_image, [[top_pad, bottom_pad], [left_pad, right_pad], [0, 0]], constant_values=0)

    return padded_image

def ImageDG_no_processed():
    return ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=lambda x: resize_and_pad_image(x, target_size=(224, 224)),
    )

def train_gen(train_dir, datagen, size, batch_size):
    return datagen.flow_from_directory(
        train_dir,
        target_size=size,  
        batch_size=batch_size,
        class_mode="categorical",
    )


def validation_gen(val_dir, datagen, size, batch_size):
    return datagen.flow_from_directory(
        val_dir,
        target_size=size,
        batch_size=batch_size,
        class_mode="categorical",
    )

def test_gen(test_dir, datagen, size, batch_size):
    return datagen.flow_from_directory(
        test_dir,
        target_size=size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )