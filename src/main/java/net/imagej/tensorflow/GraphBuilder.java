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

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

// In the fullness of time, equivalents of the methods of this class should be
// auto-generated from the OpDefs linked into libtensorflow_jni.so. That would
// match what is done in other languages like Python, C++ and Go.

public class GraphBuilder {

	private final Graph g;

	public GraphBuilder(final Graph g) {
		this.g = g;
	}

	public Output div(final Output x, final Output y) {
		return binaryOp("Div", x, y);
	}

	public Output sub(final Output x, final Output y) {
		return binaryOp("Sub", x, y);
	}

	public Output resizeBilinear(final Output images, final Output size) {
		return binaryOp("ResizeBilinear", images, size);
	}

	public Output expandDims(final Output input, final Output dim) {
		return binaryOp("ExpandDims", input, dim);
	}

	public Output constant(final String name, final Object value) {
		try (Tensor t = Tensor.create(value)) {
			return g.opBuilder("Const", name).setAttr("dtype", t.dataType())
				.setAttr("value", t).build().output(0);
		}
	}

	public Output constant(final String name, final float[] value,
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
}
