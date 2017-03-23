# Using [TensorFlow](https://www.tensorflow.org) in [ImageJ](https://imagej.net/)

Some experimentation with creating [ImageJ plugins](https://imagej.net/Writing_plugins)
that use [TensorFlow](https://www.tensorflow.org) image models.

For example, the one plugin right now pacakges the
[TensorFlow image recognition tutorial](https://www.tensorflow.org/tutorials/image_recognition),
in particular [its Java version](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java)
into a command plugin to label an opened image.

## Installation

1. Download and install [Fiji](http://fiji.sc/) - an ImageJ distribution.

2. Build these demo plugins:

   ```sh
   git checkout https://github.com/asimshankar/imagej-tensorflow
   cd imagej-tensorflow
   mvn
   ```

3. (TODO): Details on how to install the plugin

4. Explore the `TensorFlow Demo` menu.

## Caveats

The code here is a proof of concept. It has some silly inefficiencies that
would be dealt with real use. For example, the TensorFlow model is loaded
on each invocation of the command instead of being loaded once and cached
for repeated use.
