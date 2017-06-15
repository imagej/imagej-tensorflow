import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;

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
 * Command to use an Inception-image recognition model to label a JPEG image.
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

	@Parameter(type = ItemIO.OUTPUT)
	private Img<FloatType> outputImage;

	@Parameter(type = ItemIO.OUTPUT)
	private String outputLabel;

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

			final Tensor fromImgLib = loadFromImgLib(inputImage);

			try (Tensor image = constructAndExecuteGraphToNormalizeImage(
				fromImgLib))
			{
				outputImage = toImg(image);
				final float[] labelProbabilities = executeInceptionGraph(graphDef,
					image);
				final int bestLabelIdx = maxIndex(labelProbabilities);
				outputLabel = String.format("%s (%.2f%% likely)", labels.get(
					bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f);
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

	private static Img<FloatType> toImg(final Tensor image) {
		final float[] out = new float[image.numElements()];
		final Img<FloatType> tmp = ArrayImgs.floats(out, image.shape());
		image.writeTo(FloatBuffer.wrap(out));
		return tmp;
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	private static Tensor loadFromImgLib(final Dataset d) {
		return loadFromImgLib((RandomAccessibleInterval) d.getImgPlus());
	}

	private static <T extends RealType<T>> Tensor loadFromImgLib(
		final RandomAccessibleInterval<T> image)
	{
		try (final Graph g = new Graph()) {
			final GraphBuilder b = new GraphBuilder(g);
			// TODO we can be way more efficient here...
			final RandomAccess<T> source = image.randomAccess();
			final long[] dims = Intervals.dimensionsAsLongArray(image);
			final long[] reshapedDims = new long[] { dims[1], dims[0], dims[2] };

			final ArrayImg<FloatType, FloatArray> dest = ArrayImgs.floats(
				reshapedDims);
			final Cursor<FloatType> destCursor = dest.cursor();
			for (int y = 0; y < dims[1]; y++) {
				source.setPosition(y, 1);
				for (int x = 0; x < dims[0]; x++) {
					source.setPosition(x, 0);
					for (int c = 0; c < dims[2]; c++) {
						destCursor.fwd();
						source.setPosition(c, 2);
						destCursor.get().setReal(source.get().getRealDouble());
					}
				}
			}

			// Since the graph is being constructed once per execution here, we can
			// use a constant for the input image. If the graph were to be re-used for
			// multiple input images, a placeholder would have been more appropriate.
			final Output input = b.constant("input", dest.update(null)
				.getCurrentStorageArray(), reshapedDims);
			try (Session s = new Session(g)) {
				return s.runner().fetch(input.op().name()).run().get(0);
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------
	// All the code below was essentially copied verbatim from:
	// https://github.com/tensorflow/tensorflow/blob/e8f2aad0c0502fde74fc629f5b13f04d5d206700/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
	// -----------------------------------------------------------------------------------------------------------------
	private static Tensor constructAndExecuteGraphToNormalizeImage(
		final Tensor t)
	{
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

	private static int maxIndex(final float[] probabilities) {
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i) {
			if (probabilities[i] > probabilities[best]) {
				best = i;
			}
		}
		return best;
	}

	// In the fullness of time, equivalents of the methods of this class should be
	// auto-generated from the OpDefs linked into libtensorflow_jni.so. That would
	// match what is done in other languages like Python, C++ and Go.
	private static class GraphBuilder {

		GraphBuilder(final Graph g) {
			this.g = g;
		}

		Output div(final Output x, final Output y) {
			return binaryOp("Div", x, y);
		}

		Output sub(final Output x, final Output y) {
			return binaryOp("Sub", x, y);
		}

		Output resizeBilinear(final Output images, final Output size) {
			return binaryOp("ResizeBilinear", images, size);
		}

		Output expandDims(final Output input, final Output dim) {
			return binaryOp("ExpandDims", input, dim);
		}

		Output constant(final String name, final Object value) {
			try (Tensor t = Tensor.create(value)) {
				return g.opBuilder("Const", name).setAttr("dtype", t.dataType())
					.setAttr("value", t).build().output(0);
			}
		}

		Output constant(final String name, final float[] value,
			final long... shape)
		{
			try (Tensor t = Tensor.create(shape, FloatBuffer.wrap(value))) {
				return g.opBuilder("Const", name).setAttr("dtype", t.dataType())
					.setAttr("value", t).build().output(0);
			}
		}

		private Output binaryOp(final String type, final Output in1,
			final Output in2)
		{
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(
				0);
		}

		private final Graph g;
	}

	public static void main(String[] args) {
		final ImageJ ij = new ImageJ();
		ij.launch(args);
		ij.command().run(LabelImage.class, true);
	}
}
