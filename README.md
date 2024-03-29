# Neural Radiance Fields in Keras with JAX backend

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ariG23498/nerf-keras-jax/blob/master/notebooks/nerf.ipynb)

In this repository, we present a minimal JAX implementation of the research paper
[**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**](https://arxiv.org/abs/2003.08934)
by Ben Mildenhall et. al. using the Keras 3 API with JAX backend.

![loss-nerf-jax-resized](https://github.com/ariG23498/nerf-keras-jax/assets/36856589/ecb59a06-7ff4-4db0-ae04-eadafc6fb113)

Here we do the following:
- Port the [existing NeRF Keras tutorial](https://keras.io/examples/vision/nerf/) (in TensorFlow backend) from Keras-2 to Keras-3 ✨
- Utilise JAX as a backend in place of TensorFlow
- Achieve a **4X speed-up in training** compared to the TensorFlow implementation 
- Completely stateless API design

## Usage

To get started you can directly open the [notebooks/nerf.ipynb](https://github.com/ariG23498/nerf-keras-jax/blob/master/notebooks/nerf.ipynb) notebook or get started with `train.py`.

## Additional Resources

If anyone is interested in going deeper into NeRF, we have built a 3-part blog series at [PyImageSearch](https://pyimagesearch.com/).

- [Prerequisites of NeRF](https://www.pyimagesearch.com/2021/11/10/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-1/)
- [Concepts of NeRF](https://www.pyimagesearch.com/2021/11/17/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-2/)
- [Implementing NeRF](https://www.pyimagesearch.com/2021/11/24/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-3/)

## Reference

- [NeRF repository](https://github.com/bmild/nerf): The official repository for NeRF.
- [NeRF paper](https://arxiv.org/abs/2003.08934): The paper on NeRF.
- [Manim Repository](https://github.com/3b1b/manim): We have used manim to build all the animations.
- [Mathworks](https://www.mathworks.com/help/vision/ug/camera-calibration.html): Mathworks for the camera calibration article.
- [Mathew's video](https://www.youtube.com/watch?v=dPWLybp4LL0): A great video on NeRF.
