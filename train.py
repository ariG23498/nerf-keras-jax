import os

os.environ["KERAS_BACKEND"] = "jax"

from functools import partial

import keras
import tensorflow as tf

from nerf_keras_jax import callback, config, dataset, model

if __name__ == "__main__":
    AUTO = tf.data.AUTOTUNE

    data = dataset.load_data(url=config.DATASET_URL)

    images = data["images"]
    (num_images, height, width, channels) = images.shape
    (poses, focal) = (data["poses"], data["focal"])

    map_fn = partial(
        dataset.map_fn,
        height=height,
        width=width,
        focal=focal,
        num_samples=config.NUM_SAMPLES,
        pos_encode_dims=config.POS_ENCODE_DIMS,
    )

    # Create the training split.
    split_index = int(num_images * 0.8)

    # Split the images into training and validation.
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Split the poses into training and validation.
    train_poses = poses[:split_index]
    val_poses = poses[split_index:]

    # Make the training pipeline.
    train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
    train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=AUTO)
    training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
    train_ds = (
        training_ds.shuffle(config.BATCH_SIZE)
        .batch(config.BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    # Make the validation pipeline.
    val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
    val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
    val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls=AUTO)
    validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
    val_ds = (
        validation_ds.shuffle(config.BATCH_SIZE)
        .batch(config.BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    num_pos = height * width * config.NUM_SAMPLES
    nerf_model = model.get_nerf_model(
        batch_size=config.BATCH_SIZE,
        height=height,
        width=width,
        num_samples=config.NUM_SAMPLES,
        num_layers=8,
        num_pos=num_pos,
        pos_encode_dims=config.POS_ENCODE_DIMS,
    )
    nerf_model.compile(
        optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
    )

    # Create a directory to save the images during training.
    if not os.path.exists("images"):
        os.makedirs("images")
    train_ds_element = next(iter(train_ds))
    nerf_model.fit(
        train_ds,
        # validation_data=val_ds,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=[callback.TrainMonitor(train_ds_element, epochs=config.EPOCHS)],
    )
