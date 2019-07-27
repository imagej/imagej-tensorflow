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
