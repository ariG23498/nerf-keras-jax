import jax
import keras
from keras import layers, ops


class NeRF(keras.Model):
    def __init__(self, batch_size, height, width, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_samples = num_samples
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        y,
        training=False,
    ):
        rays_flat, t_vals = x
        images = y

        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            rays_flat,
            training=training,
        )

        predictions = ops.reshape(
            predictions,
            newshape=(self.batch_size, self.height, self.width, self.num_samples, 4),
        )

        # Slice the predictions into rgb and sigma.
        rgb = ops.sigmoid(predictions[..., :-1])
        sigma_a = ops.relu(predictions[..., -1])

        # Get the distance of adjacent intervals.
        delta = t_vals[..., 1:] - t_vals[..., :-1]
        delta = ops.convert_to_tensor(delta)
        delta = ops.concatenate(
            [
                delta,
                ops.broadcast_to(
                    [1e10], shape=(self.batch_size, self.height, self.width, 1)
                ),
            ],
            axis=-1,
        )
        alpha = 1.0 - ops.exp(-sigma_a * delta)

        # Get transmittance.
        exp_term = 1.0 - alpha
        epsilon = 1e-10
        transmittance = ops.cumprod(exp_term + epsilon, axis=-1)
        weights = alpha * transmittance
        rgb = ops.sum(weights[..., None] * rgb, axis=-2)

        depth_map = ops.sum(weights * t_vals, axis=-1)

        loss = self.loss_fn(images, rgb)
        return loss, (rgb, depth_map, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state

        # Get the images and the rays.
        (images, rays) = data
        (rays_flat, t_vals) = rays

        # Get the gradient function.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # Compute the gradients.
        (loss, (rgb, depth_map, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            (rays_flat, t_vals),
            images,
            training=True,
        )

        # Update trainable variables and optimizer variables.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # Update metrics.
        loss_tracker_vars = metrics_variables[: len(self.loss_tracker.variables)]
        loss_tracker_vars = self.loss_tracker.stateless_update_state(
            loss_tracker_vars, loss
        )

        logs = {}
        logs[self.loss_tracker.name] = self.loss_tracker.stateless_result(
            loss_tracker_vars
        )

        new_metrics_vars = loss_tracker_vars

        # Return metric logs and updated state variables.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker]


def get_nerf_model(
    batch_size, height, width, num_samples, num_layers, num_pos, pos_encode_dims
):
    """Generates the NeRF neural network.

    Args:
        num_layers: The number of MLP layers.
        num_pos: The number of dimensions of positional encoding.

    Returns:
        The `keras` model.
    """
    inputs = keras.Input(shape=(num_pos, 2 * 3 * pos_encode_dims + 3))
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection.
            x = layers.concatenate([x, inputs], axis=-1)
    outputs = layers.Dense(units=4)(x)
    return NeRF(
        inputs=inputs,
        outputs=outputs,
        batch_size=batch_size,
        height=height,
        width=width,
        num_samples=num_samples,
    )
