[![](https://travis-ci.org/imagej/imagej-tensorflow.svg)](https://travis-ci.org/imagej/imagej-tensorflow)

# [ImageJ](https://imagej.net/) + [TensorFlow](https://www.tensorflow.org) integration layer

This component is a library which can translate between ImageJ images and
TensorFlow tensors.

It also contains a demo ImageJ command for classifying images using a
TensorFlow image model, adapted from the [TensorFlow image recognition
tutorial](https://www.tensorflow.org/tutorials/image_recognition).

## Quickstart

```sh
git clone https://github.com/imagej/imagej-tensorflow
cd imagej-tensorflow
mvn -Pexec
```

This requires [Maven](https://maven.apache.org/install.html).  Typically `brew
install maven` on OS X, `apt-get install maven` on Ubuntu, or [detailed
instructions](https://maven.apache.org/install.html) otherwise.

## Caveats

The code here is still a proof of concept. It has some inefficiencies that
will later be improved upon. For example, the TensorFlow model is loaded
on each invocation of the command instead of being loaded once and cached
for repeated use.
