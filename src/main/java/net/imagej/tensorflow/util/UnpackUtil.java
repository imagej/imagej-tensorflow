/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2019 Board of Regents of the University of
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

package net.imagej.tensorflow.util;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.scijava.log.Logger;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Utility class for unpacking archives
 * TODO: generalize and move it to a place where others can use it as well
 *
 * @author Deborah schmidt
 */
public final class UnpackUtil {

	private UnpackUtil(){}

	public static void unGZip(String tarGzFile, String outputDir, String symLinkOutputDir, Logger logger) throws IOException {
		logger.info("Unpacking " + tarGzFile + " to " + outputDir);
		File folder = new File(outputDir);
		if (!folder.exists()) {
			folder.mkdirs();
		}

		String tarFileName = tarGzFile.replace(".gz", "");
		FileInputStream instream = new FileInputStream(tarGzFile);
		GZIPInputStream ginstream = new GZIPInputStream(instream);
		FileOutputStream outstream = new FileOutputStream(tarFileName);
		byte[] buf = new byte[1024];
		int len;
		while ((len = ginstream.read(buf)) > 0) {
			outstream.write(buf, 0, len);
		}
		ginstream.close();
		outstream.close();
		TarArchiveInputStream myTarFile = new TarArchiveInputStream(new FileInputStream(tarFileName));
		TarArchiveEntry entry;
		while ((entry = myTarFile.getNextTarEntry()) != null) {
			if (entry.isSymbolicLink() || entry.isLink()) {
				Path source = Paths.get(outputDir + entry.getName());
				Path target = Paths.get(symLinkOutputDir + entry.getLinkName());
				if (entry.isSymbolicLink()) {
					logger.info("Creating symbolic link: " + source + " -> " + target);
					Files.createSymbolicLink(source, target);
				} else {
					logger.info("Creating link: " + source + " -> " + target);
					Files.createLink(source, target);
				}
			} else {
				File output = new File(folder + "/" + entry.getName());
				if (!output.getParentFile().exists()) {
					output.getParentFile().mkdirs();
				}
				if (output.isDirectory())
					continue;
				byte[] content = new byte[(int) entry.getSize()];
				int offset = 0;
				myTarFile.read(content, offset, content.length - offset);
				logger.info("Writing " + output);
				FileOutputStream outputStream = new FileOutputStream(output);
				outputStream.write(content);
				outputStream.close();
			}
		}
		myTarFile.close();
		File tarFile = new File(tarFileName);
		tarFile.delete();
	}

	public static void unZip(String zipFile, String outputFolder, Logger logger) throws IOException {

		logger.info("Unpacking " + zipFile + " to " + outputFolder);

		byte[] buffer = new byte[1024];

		File folder = new File(outputFolder);
		if (!folder.exists()) {
			folder.mkdirs();
		}

		ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile));
		ZipEntry ze = zis.getNextEntry();

		while (ze != null) {

			String fileName = ze.getName().replace("Fiji.app/", "");
			File newFile = new File(outputFolder + File.separator + fileName);

			logger.info("Writing " + newFile.getAbsoluteFile());

			new File(newFile.getParent()).mkdirs();

			FileOutputStream fos = new FileOutputStream(newFile);

			int len;
			while ((len = zis.read(buffer)) > 0) {
				fos.write(buffer, 0, len);
			}

			fos.close();
			ze = zis.getNextEntry();

		}

		zis.closeEntry();
		zis.close();
	}

}
