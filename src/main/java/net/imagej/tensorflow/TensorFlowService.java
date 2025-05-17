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

package net.imagej.tensorflow;

import net.imagej.ImageJService;
import org.scijava.io.location.Location;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Service for working with TensorFlow.
 *
 * @author Curtis Rueden
 */
public interface TensorFlowService extends ImageJService {

	/**
	 * Extracts a persisted model from the given location.
	 * 
	 * @param source The location of the model, which must be structured as a ZIP
	 *          archive.
	 * @param modelName The name of the model by which the source should be
	 *          unpacked and cached as needed.
	 * @param tags Optional list of tags passed to
	 *          {@link SavedModelBundle#load(String, String...)}.
	 * @return The extracted TensorFlow {@link SavedModelBundle} object.
	 * @throws IOException If something goes wrong reading or unpacking the
	 *           archive.
	 * Deprecated - use {@link #loadCachedModel(Location, String, String...)} instead.
	 */
	@Deprecated
	SavedModelBundle loadModel(Location source, String modelName, String... tags)
		throws IOException;

	/**
	 * Extracts a persisted model from the given location.
	 * Returns it from cache in case it was already loaded.
	 *
	 * @param source The location of the model, which must be structured as a ZIP
	 *          archive.
	 * @param modelName The name of the model by which the source should be
	 *          unpacked and cached as needed.
	 * @param tags Optional list of tags passed to
	 *          {@link SavedModelBundle#load(String, String...)}.
	 * @return The extracted TensorFlow {@link SavedModelBundle} object
	 *           wrapped by a {@link CachedModelBundle}.
	 * @throws IOException If something goes wrong reading or unpacking the
	 *           archive.
	 */
	CachedModelBundle loadCachedModel(Location source, String modelName, String... tags)
			throws IOException;

	/**
	 * Extracts a graph from the given location.
	 * 
	 * @param source The location of the graph, which must be structured as a ZIP
	 *          archive.
	 * @param modelName The name of the model by which the source should be
	 *          unpacked and cached as needed.
	 * @param graphPath The name of the .pb file inside the ZIP archive containing
	 *          the graph.
	 * @return The extracted TensorFlow {@link Graph} object.
	 * @throws IOException If something goes wrong reading or unpacking the
	 *           archive.
	 */
	Graph loadGraph(Location source, String modelName, String graphPath)
		throws IOException;

	/**
	 * Extracts labels from the given location.
	 * 
	 * @param source The location of the labels, which must be structured as a ZIP
	 *          archive.
	 * @param modelName The name of the model by which the source should be
	 *          unpacked and cached as needed.
	 * @param labelsPath The name of the .txt file inside the ZIP archive
	 *          containing the labels.
	 * @return The extracted TensorFlow {@link Graph} object.
	 * @throws IOException If something goes wrong reading or unpacking the
	 *           archive.
	 */
	List<String> loadLabels(Location source, String modelName, String labelsPath)
		throws IOException;

	/**
	 * Loads the TensorFlow Library.
	 */
	void loadLibrary();

	/**
	 * @return the TensorFlow version which is currently loaded.
	 *         <code>null</code> if no version is loaded.
	 */
	TensorFlowVersion getTensorFlowVersion();

	/**
	 * @return Status information about the TensorFlow library (e.g. whether it crashed the application)
	 */
	TensorFlowLibraryStatus getStatus();

	/**
	 * Extracts a file from the given location.
	 * 
	 * @param source The location of the ZIP archive.
	 * @param modelName The name of the model by which the source should be
	 *          unpacked and cached as needed.
	 * @param filePath The name of the file inside the ZIP archive.
	 * @return A {@link File} object.
	 * @throws IOException If something goes wrong reading or unpacking the
	 *           archive.
	 */
	File loadFile(final Location source, final String modelName, final String filePath)
		throws IOException;
}
