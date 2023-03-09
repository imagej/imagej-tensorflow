/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2017 - 2023 Board of Regents of the University of
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

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.axis.Axes;
import net.imagej.axis.AxisType;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;
import org.scijava.command.CommandModule;

import java.util.concurrent.ExecutionException;

import static org.junit.Assert.assertNotNull;

public class LabelImageTest {

	@Test
	public void runLabelImageCommand() throws ExecutionException, InterruptedException {
		final ImageJ ij = new ImageJ();
		Dataset img = ij.dataset().create(new FloatType(), new long[]{10, 10, 3}, "", new AxisType[]{Axes.X, Axes.Y});
		CommandModule module = ij.command().run(LabelImage.class, false, "inputImage", img).get();
		assertNotNull(module);
		String labels = (String) module.getOutput("outputLabels");
		assertNotNull(labels);
		System.out.println(labels);
	}

}
