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

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.scijava.app.AppService;
import org.scijava.app.StatusService;
import org.scijava.download.DiskLocationCache;
import org.scijava.download.DownloadService;
import org.scijava.event.EventHandler;
import org.scijava.io.location.BytesLocation;
import org.scijava.io.location.Location;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.service.AbstractService;
import org.scijava.service.Service;
import org.scijava.task.Task;
import org.scijava.task.event.TaskEvent;
import org.scijava.util.ByteArray;
import org.scijava.util.FileUtils;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;

/**
 * Default implementation of {@link TensorFlowService}.
 *
 * @author Curtis Rueden
 */
@Plugin(type = Service.class)
public class DefaultTensorFlowService extends AbstractService implements
	TensorFlowService
{

	@Parameter
	private DownloadService downloadService;

	@Parameter
	private AppService appService;

	/** Models which are already cached in memory. */
	private final Map<String, SavedModelBundle> models = new HashMap<>();

	/** Graphs which are already cached in memory. */
	private final Map<String, Graph> graphs = new HashMap<>();

	/** Labels which are already cached in memory. */
	private final Map<String, List<String>> labelses = new HashMap<>();

	/** Disk cache defining where compressed models are stored locally. */
	private DiskLocationCache modelCache;

	// -- TensorFlowService methods --

	@Override
	public SavedModelBundle loadModel(final Location source,
		final String modelName, final String... tags) throws IOException
	{
		final String key = modelName + "/" + Arrays.toString(tags);

		// If the model is already cached in memory, return it.
		if (models.containsKey(key)) return models.get(key);

		// Get a local directory with unpacked model data.
		final File modelDir = modelDir(source, modelName);

		// Load the saved model.
		final SavedModelBundle model = //
			SavedModelBundle.load(modelDir.getAbsolutePath(), tags);

		return model;
	}

	@Override
	public Graph loadGraph(final Location source, final String modelName,
		final String graphPath) throws IOException
	{
		final String key = modelName + "/" + graphPath;

		// If the graph is already cached in memory, return it.
		if (graphs.containsKey(key)) return graphs.get(key);

		// Get a local directory with unpacked model data.
		final File modelDir = modelDir(source, modelName);

		// Read the serialized graph.
		final byte[] graphDef = FileUtils.readFile(new File(modelDir, graphPath));

		// Convert to a TensorFlow Graph object.
		final Graph graph = new Graph();
		graph.importGraphDef(graphDef);

		// Cache the result for performance next time.
		graphs.put(key, graph);

		return graph;
	}

	@Override
	public List<String> loadLabels(final Location source, final String modelName,
		final String labelsPath) throws IOException
	{
		final String key = modelName + "/" + labelsPath;

		// If the labels are already cached in memory, return them.
		if (labelses.containsKey(key)) return labelses.get(key);

		// Get a local directory with unpacked model data.
		final File modelDir = modelDir(source, modelName);

		// Read the labels.
		final File labelsFile = new File(modelDir, labelsPath);
		final List<String> labels;
		try (final BufferedReader labelsReader = new BufferedReader(
			new InputStreamReader(new FileInputStream(labelsFile),
				StandardCharsets.UTF_8)))
		{
			labels = labelsReader.lines().collect(Collectors.toList());
		}

		// Cache the result for performance next time.
		labelses.put(key, labels);

		return labels;
	}

	// -- Disposable methods --

	@Override
	public void dispose() {
		// Dispose models.
		for (final SavedModelBundle model : models.values()) {
			model.close();
		}
		models.clear();

		// Dispose graphs.
		for (final Graph graph : graphs.values()) {
			graph.close();
		}
		graphs.clear();

		// Dispose labels.
		labelses.clear();
	}

	// -- Helper methods --

	private DiskLocationCache modelCache() {
		if (modelCache == null) initModelCache();
		return modelCache;
	}

	private synchronized void initModelCache() {
		final DiskLocationCache cache = new DiskLocationCache();

		// Cache the models into $IMAGEJ_DIR/models.
		final File baseDir = appService.getApp().getBaseDirectory();
		final File cacheBase = new File(baseDir, "models");
		if (!cacheBase.exists()) cacheBase.mkdirs();
		cache.setBaseDirectory(cacheBase);

		modelCache = cache;
	}

	// TODO - Migrate unpacking logic into the DownloadService proper.
	// And consider whether/how to avoid using so much temporary space.

	private File modelDir(final Location source, final String modelName)
		throws IOException
	{
		final File modelDir = new File(modelCache().getBaseDirectory(), modelName);
		if (!modelDir.exists()) try {
			downloadAndUnpackResource(source, modelDir);
		}
		catch (final InterruptedException | ExecutionException exc) {
			throw new IOException(exc);
		}
		return modelDir;
	}

	/** Downloads and unpacks a zipped resource. */
	private void downloadAndUnpackResource(final Location source,
		final File destDir) throws InterruptedException, ExecutionException,
		IOException
	{
		// Allocate a dynamic byte array.
		final ByteArray byteArray = new ByteArray(1024 * 1024);

		// Download the compressed model into the byte array.
		final BytesLocation bytes = new BytesLocation(byteArray);
		final Task task = //
			downloadService.download(source, bytes, modelCache()).task();
		final StatusUpdater statusUpdater = new StatusUpdater(task);
		context().inject(statusUpdater);
		task.waitFor();

		// Extract the contents of the compressed data to the model cache.
		final byte[] buf = new byte[64 * 1024];
		final ByteArrayInputStream bais = new ByteArrayInputStream(//
			byteArray.getArray(), 0, byteArray.size());
		destDir.mkdirs();
		try (final ZipInputStream zis = new ZipInputStream(bais)) {
			while (true) {
				final ZipEntry entry = zis.getNextEntry();
				if (entry == null) break; // All done!
				final String name = entry.getName();
				statusUpdater.update("Unpacking " + name);
				final File outFile = new File(destDir, name);
				if (entry.isDirectory()) {
					outFile.mkdirs();
				}
				else {
					final int size = (int) entry.getSize();
					int len = 0;
					try (final FileOutputStream out = new FileOutputStream(outFile)) {
						while (true) {
							statusUpdater.update(len, size, "Unpacking " + name);
							final int r = zis.read(buf);
							if (r < 0) break; // end of entry
							len += r;
							out.write(buf, 0, r);
						}
					}
				}
			}
		}
		statusUpdater.clear();
	}

	/**
	 * A dumb class which passes task events on to the {@link StatusService}.
	 * Eventually, this sort of logic will be built in to SciJava Common. But for
	 * the moment, we do it ourselves.
	 */
	private class StatusUpdater {
		private final DecimalFormat formatter = new DecimalFormat("##.##");
		private final Task task;

		private long lastUpdate;

		@Parameter
		private StatusService statusService;

		private StatusUpdater(final Task task) {
			this.task = task;
		}

		public void update(final String message) {
			statusService.showStatus(message);
		}

		public void update(final int value, final int max, final String message) {
			final long timestamp = System.currentTimeMillis();
			if (timestamp < lastUpdate + 100) return; // Avoid excessive updates.
			lastUpdate = timestamp;

			final double percent = 100.0 * value / max;
			statusService.showStatus(value, max, message + ": " + //
				formatter.format(percent) + "%");
		}

		public void clear() {
			statusService.clearStatus();
		}

		@EventHandler
		private void onEvent(final TaskEvent evt) {
			if (task == evt.getTask()) {
				final int value = (int) task.getProgressValue();
				final int max = (int) task.getProgressMaximum();
				final String message = task.getStatusMessage();
				update(value, max, message);
			}
		}
	}
}
