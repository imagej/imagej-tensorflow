package net.imagej.tensorflow.lib;

import net.imagej.ImageJService;

public interface TensorFlowInstallationService extends ImageJService {

	TensorFlowLibraryVersion getCurrentVersion();

}
