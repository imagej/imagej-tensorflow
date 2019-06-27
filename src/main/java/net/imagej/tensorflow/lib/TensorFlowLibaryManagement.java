package net.imagej.tensorflow.lib;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import net.imagej.ImageJ;

@Plugin(type = Command.class, menuPath = "Plugins>TensorFlow library management")
public class TensorFlowLibaryManagement implements Command {

	// @Parameter
	private TensorFlowInstallationService tfInstallationService;

	@Override
	public void run() {
		final TensorFlowLibraryVersionsModel versionsModel = new TensorFlowLibraryVersionsModel(
				TensorFlowLibraryVersions.getAvailableLibararyVersions());
		final TensorFlowLibaryManagementFrame frame = new TensorFlowLibaryManagementFrame(versionsModel);
		// TODO add listeners
	}
	
	public static void main(final String[] args) {
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		
		ij.command().run(TensorFlowLibaryManagement.class, true);
	}
}
