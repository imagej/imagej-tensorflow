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

package net.imagej.tensorflow.options;

import java.util.Arrays;
import java.util.List;

import org.scijava.module.MutableModuleItem;
import org.scijava.options.OptionsPlugin;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

/**
 * Options for TensorFlow in ImageJ.
 * 
 * @author Curtis Rueden
 */
@Plugin(type = OptionsPlugin.class, menuPath = "Edit > Options > TensorFlow...")
public class OptionsTensorFlow extends OptionsPlugin {

	@Parameter(label = "Mode", choices = {"GPU", "CPU", "Auto"})
	private String mode = "Auto";

	@Parameter(label = "Graphics card")
	private String graphicsCard = "Auto";

	@Override
	public void initialize() {
		final MutableModuleItem<String> graphicsCardInput = //
			getInfo().getMutableInput("graphicsCard", String.class);
		graphicsCardInput.setChoices(graphicsCardChoices());
	}

	private List<String> graphicsCardChoices() {
		// FIXME: Discover available graphics cards.
		// For details on how, see:
		// https://github.com/CSBDeep/CSBDeep_website/wiki/CSBDeep-in-Fiji-%E2%80%93-Installation#multiple-gpus
		return Arrays.asList("Card1", "Card2");
	}

	public String getMode() {
		return mode;
	}

	public String getGraphicsCard() {
		return graphicsCard;
	}
}
