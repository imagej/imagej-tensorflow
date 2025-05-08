[![](https://github.com/imagej/imagej-tensorflow/actions/workflows/build-main.yml/badge.svg)](https://github.com/imagej/imagej-tensorflow/actions/workflows/build-main.yml)

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

## Issues when building from source
This project currently requires JDK 8.
You can check your Java version with:

```sh
java --version
```

For compatibility,
it is also recommended to run with TensorFlow versions 1.14 or 1.15.

If a native TensorFlow library fails to load,
a crash marker file is created at `lib/linux64/.crashed`.

If you fix the issue
(e.g. by changing the native TF library or switching Java versions),
delete the `.crashed` file, then run

```sh
mvn clean
mvn -Pexec
```