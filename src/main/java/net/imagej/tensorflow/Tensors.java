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

package net.imagej.tensorflow;

import java.nio.FloatBuffer;

import net.imglib2.Cursor;
import net.imglib2.Dimensions;
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

/**
 * Utility class for working with TensorFlow {@link Tensor} objects. In
 * particular, this class provides methods for converting between ImgLib2 data
 * structures and TensorFlow {@link Tensor}s.
 *
 * @author Curtis Rueden
 * @author Christian Dietz
 */
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
		final float[] value = floatArray(image);

		try (final Graph g = new Graph()) {
			final GraphBuilder b = new GraphBuilder(g);

			// Since the graph is being constructed once per execution here, we can
			// use a constant for the input image. If the graph were to be re-used for
			// multiple input images, a placeholder would have been more appropriate.
			final Output input = b.constant("input", value, reversedDims(image));
			try (Session s = new Session(g)) {
				return s.runner().fetch(input.op().name()).run().get(0);
			}
		}
	}

	private static long[] reversedDims(final Dimensions image) {
		final long[] dims = new long[image.numDimensions()];
		for (int d = 0; d < dims.length; d++) {
			dims[dims.length - d - 1] = image.dimension(d);
		}
		return dims;
	}

	private static <T extends RealType<T>> float[] floatArray(
		final RandomAccessibleInterval<T> image)
	{
		// TODO we can be way more efficient here...
		final RandomAccess<T> source = image.randomAccess();
		final long[] dims = Intervals.dimensionsAsLongArray(image);

		final ArrayImg<FloatType, FloatArray> dest = ArrayImgs.floats(dims);
		final Cursor<FloatType> destCursor = dest.localizingCursor();
		while (destCursor.hasNext()) {
			destCursor.fwd();
			source.setPosition(destCursor);
			destCursor.get().setReal(source.get().getRealDouble());
		}
		return dest.update(null).getCurrentStorageArray();
	}
}
