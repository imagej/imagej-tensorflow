package net.imagej.tensorflow.lib;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

class TensorFlowLibraryVersionsModel {

	private static final String NO_FILTER = "-";

	private static final String GPU_FILTER = "GPU";

	private static final String CPU_FILTER = "CPU";

	private final List<TensorFlowLibraryVersion> availableVersions;

	private String gpuFilter = NO_FILTER;

	private String cudaFilter = NO_FILTER;

	private String tfFilter = NO_FILTER;

	TensorFlowLibraryVersionsModel(final List<TensorFlowLibraryVersion> availableVersions) {
		this.availableVersions = availableVersions;
	}

	String[] getGPUFilterChoices() {
		return new String[] { NO_FILTER, GPU_FILTER, CPU_FILTER };
	}

	String[] getCudaFilterChoices() {
		return getFilterChoicesFrom(availableVersions, TensorFlowLibraryVersion::getCudaVersion);
	}

	String[] getTFFilterChoices() {
		return getFilterChoicesFrom(availableVersions, v -> Optional.of(v.getTfVersion()));
	}

	void setGpuFilter(final String gpuFilter) {
		this.gpuFilter = gpuFilter;
	}

	void setCudaFilter(final String cudaFilter) {
		this.cudaFilter = cudaFilter;
	}

	void setTfFilter(final String tfFilter) {
		this.tfFilter = tfFilter;
	}

	List<TensorFlowLibraryVersion> getFilteredVersions() {
		return availableVersions.stream().filter(v -> isTFVersionInFilter(v)).collect(Collectors.toList());
	}

	private boolean isTFVersionInFilter(final TensorFlowLibraryVersion v) {
		// GPU or CPU
		if ((gpuFilter.equals(GPU_FILTER) && !v.isUsesGPU()) || //
				(gpuFilter.equals(CPU_FILTER) && v.isUsesGPU())) {
			return false;
		}
		// CUDA version
		if (!cudaFilter.equals(NO_FILTER) && //
				v.getCudaVersion().isPresent() && //
				!cudaFilter.equals(v.getCudaVersion().get())) {
			return false;
		}
		// TF version
		if (!tfFilter.equals(NO_FILTER) && //
				!tfFilter.equals(v.getTfVersion())) {
			return false;
		}
		// Not filtered out
		return true;
	}

	private static <T> String[] getFilterChoicesFrom(final Collection<T> available,
			final Function<T, Optional<String>> getChoiceFn) {
		final Set<String> choices = new LinkedHashSet<>();
		choices.add(NO_FILTER);
		for (final T i : available) {
			final Optional<String> choice = getChoiceFn.apply(i);
			if (choice.isPresent()) {
				choices.add(choice.get());
			}
		}
		return choices.toArray(new String[0]);
	}
}
