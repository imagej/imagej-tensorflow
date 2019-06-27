package net.imagej.tensorflow.lib;

import java.util.ArrayList;
import java.util.List;

import net.imagej.updater.util.UpdaterUtil;

public class TensorFlowLibraryVersions {

	private TensorFlowLibraryVersions() {
		// Utility class
	}

	public static List<TensorFlowLibraryVersion> getAvailableLibararyVersions() {
		final String platform = UpdaterUtil.getPlatform();
		if (platform.equals("linux64")) {
			return getLinuxVersions();
		}
		// TODO
		return null;
	}
	
	private static List<TensorFlowLibraryVersion> getLinuxVersions() {
		final List<TensorFlowLibraryVersion> versions = new ArrayList<>();
		versions.add( TensorFlowLibraryVersion.cpuOf("1.2.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.3.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.4.1"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.5.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.6.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.7.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.8.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.9.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.10.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.11.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.12.0"));
		versions.add( TensorFlowLibraryVersion.cpuOf("1.13.1"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.2.0", "8.0", "5.1"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.3.0", "8.0", "6"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.4.1", "8.0", "6"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.5.0", "9.0", "7"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.6.0", "9.0", "7"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.7.0", "9.0", "7"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.8.0", "9.0", "7"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.9.0", "9.0", "7"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.10.0", "9.0", "7.?"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.11.0", "9.0", "7.2"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.12.0", "9.0", "7.2"));
		versions.add( TensorFlowLibraryVersion.gpuOf("1.13.1", "10.0", "7.4"));
		return versions;
	}
}
