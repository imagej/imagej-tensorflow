/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2017 Board of Regents of the University of
 * Wisconsin-Madison and Google, Inc.
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

package net.imagej.tensorflow.demo;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.tensorflow.GraphBuilder;
import net.imagej.tensorflow.Tensors;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * Command to use an Inception-image recognition model to label an image.
 * <p>
 * See the
 * <a href="https://www.tensorflow.org/tutorials/image_recognition">TensorFlow
 * image recognition tutorial</a> and its <a href=
 * "www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java">
 * Java implementation</a>.
 */
@Plugin(type = Command.class, menuPath = "TensorFlow Demos>Label Image",
	headless = true)
public class LabelImage implements Command {

	@Parameter
	private LogService log;

	@Parameter
	private Dataset inputImage;

	@Parameter(label = "Min probability (%)", min = "0", max = "100")
	private double minPercent = 1;

	@Parameter(type = ItemIO.OUTPUT)
	private Img<FloatType> outputImage;

	@Parameter(type = ItemIO.OUTPUT)
	private String outputLabels;

	@Override
	public void run() {
		try {
			// This is not efficient: Loading the model, the labels and constructing a
			// graph to normalize the image on every call to run(). Fine for a
			// proof-of-concept demo, but in any real implementation, model loading
			// should be ammortized.
			final byte[] graphDef = loadInceptionModelGraphDef();
			final List<String> labels = loadInceptionLabels();
			log.info("Loaded GraphDef of " + graphDef.length + " bytes and " + labels
				.size() + " labels");

			final Tensor inputTensor = loadFromImgLib(inputImage);

			try (final Tensor image = normalizeImage(inputTensor)) {
				outputImage = Tensors.img(image);
				final float[] labelProbabilities = executeInceptionGraph(graphDef,
					image);

				// Sort labels by probability.
				final int labelCount = Math.min(labelProbabilities.length, labels
					.size());
				final Integer[] labelIndices = IntStream.range(0, labelCount).boxed()
					.toArray(Integer[]::new);
				Arrays.sort(labelIndices, 0, labelCount, new Comparator<Integer>() {

					@Override
					public int compare(final Integer i1, final Integer i2) {
						final float p1 = labelProbabilities[i1];
						final float p2 = labelProbabilities[i2];
						return p1 == p2 ? 0 : p1 > p2 ? -1 : 1;
					}
				});

				// Output labels above the probability threshold.
				final double cutoff = minPercent / 100; // % -> probability
				final StringBuilder sb = new StringBuilder();
				for (int i = 0; i < labelCount; i++) {
					final int index = labelIndices[i];
					final double p = labelProbabilities[index];
					if (p < cutoff) break;

					sb.append(String.format("%s (%.2f%% likely)\n", labels.get(index),
						labelProbabilities[index] * 100f));
				}
				outputLabels = sb.toString();
			}
		}
		catch (final Exception exc) {
			log.error(exc);
		}
	}

	private byte[] loadInceptionModelGraphDef() throws IOException {
		final String path = Paths.get("tensorflow_models", "inception5h",
			"tensorflow_inception_graph.pb").toString();
		getClass().getClassLoader();
		final int nbytes = ClassLoader.getSystemResource(path).openConnection()
			.getContentLength();
		final byte[] graphDef = new byte[nbytes];
		log.info("Reading " + nbytes + " bytes of the TensorFlow inception model");
		try (DataInputStream is = new DataInputStream(getClass().getClassLoader()
			.getResourceAsStream(path)))
		{
			is.readFully(graphDef);
		}
		return graphDef;
	}

	private List<String> loadInceptionLabels() throws IOException {
		final String path = Paths.get("tensorflow_models", "inception5h",
			"imagenet_comp_graph_label_strings.txt").toString();
		try (InputStream is = getClass().getClassLoader().getResourceAsStream(
			path))
		{
			return new BufferedReader(new InputStreamReader(is,
				StandardCharsets.UTF_8)).lines().collect(Collectors.toList());
		}
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	private static Tensor loadFromImgLib(final Dataset d) {
		return loadFromImgLib((RandomAccessibleInterval) d.getImgPlus());
	}

	private static <T extends RealType<T>> Tensor loadFromImgLib(
		final RandomAccessibleInterval<T> image)
	{
		// NB: Assumes XYC ordering. TensorFlow wants CXY.
		// TODO: Need to be smarter about this.
		final IntervalView<T> imageCXY = //
			Views.permute(Views.permute(image, 2, 1), 0, 1);
		return Tensors.tensor(imageCXY);
	}

	// -----------------------------------------------------------------------------------------------------------------
	// All the code below was essentially copied verbatim from:
	// https://github.com/tensorflow/tensorflow/blob/e8f2aad0c0502fde74fc629f5b13f04d5d206700/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
	// -----------------------------------------------------------------------------------------------------------------
	private static Tensor normalizeImage(final Tensor t) {
		try (Graph g = new Graph()) {
			final GraphBuilder b = new GraphBuilder(g);
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to
			//   float using (value - Mean)/Scale.
			final int H = 224;
			final int W = 224;
			final float mean = 117f;
			final float scale = 1f;

			// Since the graph is being constructed once per execution here, we can
			// use a constant for the input image. If the graph were to be re-used for
			// multiple input images, a placeholder would have been more appropriate.
			final Output input = g.opBuilder("Const", "input")//
				.setAttr("dtype", t.dataType())//
				.setAttr("value", t).build().output(0);
			final Output output = b.div(b.sub(b.resizeBilinear(b.expandDims(//
				input, //
				b.constant("make_batch", 0)), //
				b.constant("size", new int[] { H, W })), //
				b.constant("mean", mean)), //
				b.constant("scale", scale));
			try (Session s = new Session(g)) {
				return s.runner().fetch(output.op().name()).run().get(0);
			}
		}
	}

	private static float[] executeInceptionGraph(final byte[] graphDef,
		final Tensor image)
	{
		try (Graph g = new Graph()) {
			g.importGraphDef(graphDef);
			try (Session s = new Session(g);
					Tensor result = s.runner().feed("input", image).fetch("output").run()
						.get(0))
			{
				final long[] rshape = result.shape();
				if (result.numDimensions() != 2 || rshape[0] != 1) {
					throw new RuntimeException(String.format(
						"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
						Arrays.toString(rshape)));
				}
				final int nlabels = (int) rshape[1];
				return result.copyTo(new float[1][nlabels])[0];
			}
		}
	}

	public static void main(String[] args) throws IOException {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		final String imagePath = "http://samples.fiji.sc/new-lenna.jpg";
		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		// Launch the "Label Images" command with some sensible defaults.
		ij.command().run(LabelImage.class, true, //
			"inputImage", dataset);
	}
}
