package net.imagej.tensorflow;

import net.imagej.tensorflow.util.TensorFlowUtil;
import org.junit.Test;
import org.tensorflow.TensorFlow;

import java.net.URL;

import static org.junit.Assert.assertEquals;

public class VersionBuilderTest {
	@Test
	public void testJARVersion() {
		URL source = TensorFlow.class.getResource("TensorFlow.class");
		TensorFlowVersion version = TensorFlowUtil.getTensorFlowJARVersion(source);
		assertEquals(TensorFlow.version(), version.getVersionNumber());
	}
}
