# Analysis in [ImageJ](https://imagej.net/)/[Fiji](http://fiji.sc) using [TensorFlow](https://www.tensorflow.org) models

[![Build Status](https://travis-ci.org/asimshankar/imagej-tensorflow.svg)](https://travis-ci.org/asimshankar/imagej-tensorflow)

Some experimentation with creating [ImageJ plugins](https://imagej.net/Writing_plugins)
that use [TensorFlow](https://www.tensorflow.org) image models.

For example, the one plugin right now pacakges the
[TensorFlow image recognition tutorial](https://www.tensorflow.org/tutorials/image_recognition),
in particular [its Java version](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java)
into a command plugin to label an opened image.

## Quickstart

```sh
git clone https://github.com/asimshankar/imagej-tensorflow
cd imagej-tensorflow
mvn compile exec:java
```

This requires [Maven](https://maven.apache.org/install.html).  Typically `brew
install maven` on OS X, `apt-get install maven` on Ubuntu, or [detailed
instructions](https://maven.apache.org/install.html) otherwise.

## Caveats

The code here is a proof of concept. It has some silly inefficiencies that
would be dealt with real use. For example, the TensorFlow model is loaded
on each invocation of the command instead of being loaded once and cached
for repeated use.
