/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2017 - 2025 Board of Regents of the University of
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
package net.imagej.tensorflow.ui;

import net.imagej.tensorflow.TensorFlowVersion;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class TensorFlowVersionTest {

	@Test
	public void compareVersions() {
		TensorFlowVersion v1 = new TensorFlowVersion("1.4.0", true, null, null);
		TensorFlowVersion v2 = new TensorFlowVersion("1.4.0", true, null, null);
		assertEquals(v1, v2);

		v1 = new TensorFlowVersion("1.4.0", true, null, null);
		v2 = new TensorFlowVersion("1.4.0", false, null, null);
		assertNotEquals(v1, v2);

		v1 = new TensorFlowVersion("1.4.0", false, null, null);
		v2 = new TensorFlowVersion("1.4.0", false, "x.x", "y.y");
		assertEquals(v1, v2);

		DownloadableTensorFlowVersion v3 = new DownloadableTensorFlowVersion("1.4.0", false);
		assertNotEquals(v3, v1);
		assertNotEquals(v1, v3);

	}

}
