import keras
import numpy as np
from matplotlib import pyplot as plt


class TrainMonitor(keras.callbacks.Callback):
    def __init__(self, train_ds_element, epochs):
        # test_imgs, test_rays = next(iter(train_ds))
        self.test_imgs, self.test_rays = train_ds_element
        self.test_rays_flat, self.test_t_vals = self.test_rays
        self.epochs = epochs
        self.loss_list = list()

    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        self.loss_list.append(loss)

        trainable_variables = self.model.trainable_variables
        non_trainable_variables = self.model.non_trainable_variables

        loss, (test_recons_images, depth_maps, non_trainable_variables) = (
            self.model.compute_loss_and_updates(
                trainable_variables,
                non_trainable_variables,
                (self.test_rays_flat, self.test_t_vals),
                self.test_imgs,
                training=False,
            )
        )

        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.utils.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.utils.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        ax[2].plot(self.loss_list)
        ax[2].set_xticks(np.arange(0, self.epochs + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(f"images/{epoch:03d}.png")
        plt.show()
        plt.close()
