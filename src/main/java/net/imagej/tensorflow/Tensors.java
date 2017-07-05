package net.imagej.tensorflow;

import java.nio.FloatBuffer;

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

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public final class Tensors {

	private Tensors() {
		// NB: Prevent instantiation of utility class.
	}

	public static Img<FloatType> img(final Tensor image) {
		final float[] out = new float[image.numElements()];
		final Img<FloatType> tmp = ArrayImgs.floats(out, image.shape());
		image.writeTo(FloatBuffer.wrap(out));
		return tmp;
	}

	public static <T extends RealType<T>> Tensor tensor(
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
}
