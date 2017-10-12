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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

import org.junit.Test;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;

import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;

public class TensorsTest {

	// --------- TENSOR to RAI ---------

	/** Tests the img<Type>(Tensor) functions */
	@Test
	public void testTensorToImgReverse() {
		final long[] shape = new long[] { 20, 10, 3 };
		final int[] mapping = new int[] { 2, 1, 0 };
		final long[] dims = new long[] { 3, 10, 20 };
		final int n = shape.length;
		final int size = 600;

		// Get some points to mark
		List<Point> points = createTestPoints(n);

		// Create Tensors of different type and convert them to images
		ByteBuffer dataByte = ByteBuffer.allocateDirect(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataByte.put(i, (byte) (int) v));
		Tensor tensorByte = Tensor.create(DataType.UINT8, shape, dataByte);
		Img<ByteType> imgByte = Tensors.imgByte(tensorByte);

		DoubleBuffer dataDouble = DoubleBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataDouble.put(i, v));
		Tensor tensorDouble = Tensor.create(shape, dataDouble);
		Img<DoubleType> imgDouble = Tensors.imgDouble(tensorDouble);

		FloatBuffer dataFloat = FloatBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataFloat.put(i, v));
		Tensor tensorFloat = Tensor.create(shape, dataFloat);
		Img<FloatType> imgFloat = Tensors.imgFloat(tensorFloat);

		IntBuffer dataInt = IntBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataInt.put(i, v));
		Tensor tensorInt = Tensor.create(shape, dataInt);
		Img<IntType> imgInt = Tensors.imgInt(tensorInt);

		LongBuffer dataLong = LongBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataLong.put(i, v));
		Tensor tensorLong = Tensor.create(shape, dataLong);
		Img<LongType> imgLong = Tensors.imgLong(tensorLong);

		// Check all created images
		checkImage(imgByte, n, dims, points);
		checkImage(imgDouble, n, dims, points);
		checkImage(imgFloat, n, dims, points);
		checkImage(imgInt, n, dims, points);
		checkImage(imgLong, n, dims, points);
	}

	/** Tests the img<Type>Direct(Tensor) functions */
	@Test
	public void testTensorToImgDirect() {
		final long[] shape = new long[] { 20, 10, 3 };
		final int[] mapping = new int[] { 0, 1, 2 };
		final int n = shape.length;
		final int size = 600;

		// Get some points to mark
		List<Point> points = createTestPoints(n);

		// Create Tensors of different type and convert them to images
		ByteBuffer dataByte = ByteBuffer.allocateDirect(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataByte.put(i, (byte) (int) v));
		Tensor tensorByte = Tensor.create(DataType.UINT8, shape, dataByte);
		Img<ByteType> imgByte = Tensors.imgByteDirect(tensorByte);

		DoubleBuffer dataDouble = DoubleBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataDouble.put(i, v));
		Tensor tensorDouble = Tensor.create(shape, dataDouble);
		Img<DoubleType> imgDouble = Tensors.imgDoubleDirect(tensorDouble);

		FloatBuffer dataFloat = FloatBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataFloat.put(i, v));
		Tensor tensorFloat = Tensor.create(shape, dataFloat);
		Img<FloatType> imgFloat = Tensors.imgFloatDirect(tensorFloat);

		IntBuffer dataInt = IntBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataInt.put(i, v));
		Tensor tensorInt = Tensor.create(shape, dataInt);
		Img<IntType> imgInt = Tensors.imgIntDirect(tensorInt);

		LongBuffer dataLong = LongBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataLong.put(i, v));
		Tensor tensorLong = Tensor.create(shape, dataLong);
		Img<LongType> imgLong = Tensors.imgLongDirect(tensorLong);

		// Check all created images
		checkImage(imgByte, n, shape, points);
		checkImage(imgDouble, n, shape, points);
		checkImage(imgFloat, n, shape, points);
		checkImage(imgInt, n, shape, points);
		checkImage(imgLong, n, shape, points);
	}

	/** Tests the img<Type>(Tensor, int[]) functions */
	@Test
	public void testTensorToImgMapping() {
		final long[] shape = new long[] { 3, 5, 2, 4 };
		final int[] mapping = new int[] { 1, 3, 0, 2 }; // A strange mapping
		final long[] dims = new long[] { 5, 4, 3, 2 };
		final int n = shape.length;
		final int size = 120;

		// Get some points to mark
		List<Point> points = createTestPoints(n);

		// Create Tensors of different type and convert them to images
		ByteBuffer dataByte = ByteBuffer.allocateDirect(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataByte.put(i, (byte) (int) v));
		Tensor tensorByte = Tensor.create(DataType.UINT8, shape, dataByte);
		Img<ByteType> imgByte = Tensors.imgByte(tensorByte, mapping);

		DoubleBuffer dataDouble = DoubleBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataDouble.put(i, v));
		Tensor tensorDouble = Tensor.create(shape, dataDouble);
		Img<DoubleType> imgDouble = Tensors.imgDouble(tensorDouble, mapping);

		FloatBuffer dataFloat = FloatBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataFloat.put(i, v));
		Tensor tensorFloat = Tensor.create(shape, dataFloat);
		Img<FloatType> imgFloat = Tensors.imgFloat(tensorFloat, mapping);

		IntBuffer dataInt = IntBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataInt.put(i, v));
		Tensor tensorInt = Tensor.create(shape, dataInt);
		Img<IntType> imgInt = Tensors.imgInt(tensorInt, mapping);

		LongBuffer dataLong = LongBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataLong.put(i, v));
		Tensor tensorLong = Tensor.create(shape, dataLong);
		Img<LongType> imgLong = Tensors.imgLong(tensorLong, mapping);

		// Check all created images
		checkImage(imgByte, n, dims, points);
		checkImage(imgDouble, n, dims, points);
		checkImage(imgFloat, n, dims, points);
		checkImage(imgInt, n, dims, points);
		checkImage(imgLong, n, dims, points);
	}

	/** Checks one image for the dimensions and marked points */
	private <T extends RealType<T>> void checkImage(final Img<T> img, final int n, final long[] dims,
			final List<Point> points) {
		assertArrayEquals(dims, Intervals.dimensionsAsLongArray(img));
		assertEquals(n, img.numDimensions());
		checkPoints(img, points);
	}

	// --------- RAI to TENSOR ---------

	/** Tests the tensor(RAI) function */
	@Test
	public void testImgToTensorReverse() {
		assertEquals(1, 1);

		final long[] dims = new long[] { 20, 10, 3 };
		final long[] shape = new long[] { 3, 10, 20 };
		final int n = dims.length;

		// ByteType
		testImg2TensorReverseForType(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), n, shape,
				DataType.UINT8);

		// DoubleType
		testImg2TensorReverseForType(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), n, shape,
				DataType.DOUBLE);

		// FloatType
		testImg2TensorReverseForType(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), n, shape,
				DataType.FLOAT);

		// IntType
		testImg2TensorReverseForType(new ArrayImgFactory<IntType>().create(dims, new IntType()), n, shape,
				DataType.INT32);

		// LongType
		testImg2TensorReverseForType(new ArrayImgFactory<LongType>().create(dims, new LongType()), n, shape,
				DataType.INT64);
	}

	/** Tests the tensorDirect(RAI) function */
	@Test
	public void testImgToTensorDirect() {
		assertEquals(1, 1);

		final long[] dims = new long[] { 20, 10, 3 };
		final int n = dims.length;

		// ByteType
		testImg2TensorDirectForType(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), n, dims,
				DataType.UINT8);

		// DoubleType
		testImg2TensorDirectForType(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), n, dims,
				DataType.DOUBLE);

		// FloatType
		testImg2TensorDirectForType(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), n, dims,
				DataType.FLOAT);

		// IntType
		testImg2TensorDirectForType(new ArrayImgFactory<IntType>().create(dims, new IntType()), n, dims,
				DataType.INT32);

		// LongType
		testImg2TensorDirectForType(new ArrayImgFactory<LongType>().create(dims, new LongType()), n, dims,
				DataType.INT64);
	}

	/** Tests the tensor(RAI, int[]) function */
	@Test
	public void testImgToTensorMapping() {
		assertEquals(1, 1);

		final long[] dims = new long[] { 5, 4, 3, 2 };
		final int[] mapping = new int[] { 1, 3, 0, 2 }; // A strange mapping
		final long[] shape = new long[] { 3, 5, 2, 4 };
		final int n = dims.length;

		// ByteType
		testImg2TensorMappingForType(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), mapping, n, shape,
				DataType.UINT8);

		// DoubleType
		testImg2TensorMappingForType(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), mapping, n,
				shape, DataType.DOUBLE);

		// FloatType
		testImg2TensorMappingForType(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), mapping, n, shape,
				DataType.FLOAT);

		// IntType
		testImg2TensorMappingForType(new ArrayImgFactory<IntType>().create(dims, new IntType()), mapping, n, shape,
				DataType.INT32);

		// LongType
		testImg2TensorMappingForType(new ArrayImgFactory<LongType>().create(dims, new LongType()), mapping, n, shape,
				DataType.INT64);
	}

	/** Tests the tensor(RAI) function for one image */
	private <T extends RealType<T>> void testImg2TensorReverseForType(final Img<T> img, final int n, final long[] shape,
			final DataType t) {
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);

		Tensor tensor = Tensors.tensor(img);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(t, tensor.dataType());
		checkPointsTensor(tensor, IntStream.range(0, n).map(i -> n - 1 - i).toArray(), points);
	}

	/** Tests the tensorDirect(RAI) function for one image */
	private <T extends RealType<T>> void testImg2TensorDirectForType(final Img<T> img, final int n, final long[] shape,
			final DataType t) {
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);

		Tensor tensor = Tensors.tensorDirect(img);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(t, tensor.dataType());
		checkPointsTensor(tensor, IntStream.range(0, n).toArray(), points);
	}

	/** Tests the tensor(RAI, int[]) function for one image */
	private <T extends RealType<T>> void testImg2TensorMappingForType(final Img<T> img, final int[] mapping,
			final int n, final long[] shape, final DataType t) {
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);
		Tensor tensor = Tensors.tensor(img, mapping);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(t, tensor.dataType());
		checkPointsTensor(tensor, mapping, points);
	}

	/** Creates some interesting points to mark */
	private List<Point> createTestPoints(int n) {
		List<Point> points = new ArrayList<>();
		points.add(new Point(n));
		for (int d = 0; d < n; d++) {
			Point p = new Point(n);
			p.fwd(d);
			points.add(p);
		}
		return points;
	}

	/** Marks a list of points in an image */
	private <T extends RealType<T>> void markPoints(final Img<T> img, final List<Point> points) {
		for (int i = 0; i < points.size(); i++) {
			RandomAccess<T> randomAccess = img.randomAccess();
			randomAccess.setPosition(points.get(i));
			randomAccess.get().setReal(i + 1);
		}
	}

	/** Checks if points in an image are set to the right value */
	private <T extends RealType<T>> void checkPoints(final Img<T> img, List<Point> points) {
		for (int i = 0; i < points.size(); i++) {
			RandomAccess<T> randomAccess = img.randomAccess();
			randomAccess.setPosition(points.get(i));
			assertEquals(i + 1, randomAccess.get().getRealFloat(), 0.001);
		}
	}

	/**
	 * Calculates the index of a point in a buffer of a Tensor of the given
	 * shape
	 */
	private int calcIndex(final long[] shape, final int[] mapping, final Point p) {
		assert p.numDimensions() == shape.length;
		int n = shape.length;

		// switch index and value of the mapping
		int[] switchedMapping = new int[mapping.length];
		IntStream.range(0, mapping.length).forEach(i -> switchedMapping[mapping[i]] = i);

		int index = 0;
		for (int dimPreSize = 1, d = n - 1; d >= 0; d--) {
			index += dimPreSize * p.getIntPosition(switchedMapping[d]);
			dimPreSize *= shape[d];
		}
		return index;
	}

	/** Checks the given points for the given tensor */
	private void checkPointsTensor(final Tensor tensor, final int[] mapping, final List<Point> points) {
		switch (tensor.dataType()) {
		case UINT8: {
			ByteBuffer buffer = ByteBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case DOUBLE: {
			DoubleBuffer buffer = DoubleBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case FLOAT: {
			FloatBuffer buffer = FloatBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case INT32: {
			IntBuffer buffer = IntBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case INT64: {
			LongBuffer buffer = LongBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		default:
			// This should not happen because the type is checked before
			fail("Tensor has unsupported type.");
		}
	}

	/**
	 * Calculates the index of a point in a Tensor buffer and executes the
	 * BiConsumer with the index and the value of the point (which is the index
	 * in the list + 1).
	 */
	private void execForPointsWithBufferIndex(final long[] shape, final int[] mapping, final List<Point> points,
			final BiConsumer<Integer, Integer> exec) {
		for (int i = 0; i < points.size(); i++) {
			exec.accept(calcIndex(shape, mapping, points.get(i)), i + 1);
		}
	}
}
