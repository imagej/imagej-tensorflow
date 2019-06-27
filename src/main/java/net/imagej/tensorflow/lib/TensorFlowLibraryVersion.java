package net.imagej.tensorflow.lib;

import java.util.Objects;
import java.util.Optional;

/**
 * TODO Should this be called somehow else? Not only the version but also if it
 * is the GPU edition and which platform it is for
 */
public final class TensorFlowLibraryVersion {

	private static final String CPU_TEXT = "CPU";

	private static final String GPU_TEXT = "GPU";

	private final String tfVersion;

	private final boolean usesGPU;

	private final Optional<String> cudaVersion;

	private final Optional<String> cudnnVersion;

	public static TensorFlowLibraryVersion cpuOf(final String tfVersion) {
		return new TensorFlowLibraryVersion(tfVersion);
	}

	public static TensorFlowLibraryVersion gpuOf(final String tfVersion, final String cudaVersion,
			final String cudnnVersion) {
		return new TensorFlowLibraryVersion(tfVersion, cudaVersion, cudnnVersion);
	}

	/** Creates a GPU library version */
	private TensorFlowLibraryVersion(final String tfVersion, final String cudaVersion, final String cudnnVersion) {
		this.tfVersion = tfVersion;
		this.usesGPU = true;
		this.cudaVersion = Optional.of(cudaVersion);
		this.cudnnVersion = Optional.of(cudnnVersion);
	}

	/** Creates a CPU library version */
	private TensorFlowLibraryVersion(final String tfVersion) {
		this.tfVersion = tfVersion;
		this.usesGPU = false;
		this.cudaVersion = Optional.empty();
		this.cudnnVersion = Optional.empty();
	}

	public String getTfVersion() {
		return tfVersion;
	}

	public boolean isUsesGPU() {
		return usesGPU;
	}

	public Optional<String> getCudaVersion() {
		return cudaVersion;
	}

	public Optional<String> getCudnnVersion() {
		return cudnnVersion;
	}

	@Override
	public boolean equals(final Object obj) {
		if (!(obj instanceof TensorFlowLibraryVersion)) {
			// Note: Handles if obj is null
			return false;
		}
		final TensorFlowLibraryVersion o = (TensorFlowLibraryVersion) obj;
		return tfVersion.equals(o.tfVersion) && //
				usesGPU == o.usesGPU && //
				cudaVersion.equals(o.cudaVersion) && //
				cudnnVersion.equals(o.cudnnVersion);
	}

	@Override
	public int hashCode() {
		return Objects.hash(tfVersion, usesGPU, cudaVersion, cudnnVersion);
	}

	@Override
	public String toString() {
		final StringBuilder builder = new StringBuilder();
		builder.append("TF ").append(tfVersion).append(" ");
		builder.append(usesGPU ? GPU_TEXT : CPU_TEXT).append(" ");
		if (usesGPU) {
			builder.append("(");
			builder.append("CUDA ").append(cudaVersion.get());
			builder.append(", ");
			builder.append("cuDNN ").append(cudnnVersion.get());
			builder.append(")");
		}
		// TODO handle downloaded
		return builder.toString();
	}
}
