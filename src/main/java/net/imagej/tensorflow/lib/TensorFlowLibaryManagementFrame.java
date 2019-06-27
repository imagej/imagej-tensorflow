package net.imagej.tensorflow.lib;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;

import net.miginfocom.swing.MigLayout;

public class TensorFlowLibaryManagementFrame extends JFrame {

	private static final long serialVersionUID = 1L;

	private static final Color LIST_BACKGROUND_COLOR = new Color(250, 250, 250);

	private final TensorFlowLibraryVersionsModel versionsModel;

	private JPanel installPanel;

	private JComboBox<String> gpuChoiceBox;

	private JComboBox<String> cudaChoiceBox;

	private JComboBox<String> tfChoiceBox;

	/** Creates and shows a new frame for selecting a TensorFlow library version. */
	TensorFlowLibaryManagementFrame(final TensorFlowLibraryVersionsModel versionsModel) {
		// TODO title with upper case letters?
		super("TensorFlow library version management");
		this.versionsModel = versionsModel;

		init();
		pack();
		setLocationRelativeTo(null);
		setMaximumSize(new Dimension(0, 200));
		setVisible(true);
		// TODO update status?
	}

	private void init() {
		JPanel panel = new JPanel();
		panel.setLayout(new MigLayout("height 400"));
		panel.add(new JLabel("Please select the TensorFlow version you would like to install."), "wrap");
		panel.add(createFilterPanel(), "wrap");
		panel.add(createInstallPanel(), "wrap, span, grow");
		panel.add(createStatusPanel(), "span, grow");
		setContentPane(panel);
	}

	private Component createFilterPanel() {
		// Select GPU or CPU
		final String[] gpuChoices = versionsModel.getGPUFilterChoices();
		gpuChoiceBox = new JComboBox<>(gpuChoices);
		gpuChoiceBox.addActionListener(e -> {
			versionsModel.setGpuFilter((String) gpuChoiceBox.getSelectedItem());
			updateFilter();
		});

		// Select CUDA version
		final String[] cudaChoices = versionsModel.getCudaFilterChoices();
		cudaChoiceBox = new JComboBox<>(cudaChoices);
		cudaChoiceBox.addActionListener(e -> {
			versionsModel.setCudaFilter((String) cudaChoiceBox.getSelectedItem());
			updateFilter();
		});

		// Select TensorFlow version
		final String[] tfChoices = versionsModel.getTFFilterChoices();
		tfChoiceBox = new JComboBox<>(tfChoices);
		tfChoiceBox.addActionListener(e -> {
			versionsModel.setTfFilter((String) tfChoiceBox.getSelectedItem());
			updateFilter();
		});

		final JPanel panel = new JPanel();
		panel.setLayout(new MigLayout());
		panel.add(createLabel("Filter by.."));
		panel.add(createLabel("Mode: "));
		panel.add(gpuChoiceBox);
		panel.add(createLabel("CUDA: "));
		panel.add(cudaChoiceBox);
		panel.add(createLabel("TensorFlow: "));
		panel.add(tfChoiceBox);
		panel.setBackground(LIST_BACKGROUND_COLOR);
		// TODO remove commented code
//		panel.setBorder(BorderFactory.createMatteBorder(1, 0, 0, 0, Color.darkGray));
		return panel;
	}

	private Component createInstallPanel() {
		installPanel = new JPanel(new MigLayout("flowy"));
		final JScrollPane scroll = new JScrollPane(installPanel);
		scroll.setBorder(BorderFactory.createEmptyBorder());
		installPanel.setBackground(LIST_BACKGROUND_COLOR);
		return scroll;
	}

	private void updateInstallChoices() {
		// TODO Add buttons to the install panel according to the filter
		final ButtonGroup versionGroup = new ButtonGroup();
		installPanel.removeAll();
		for (final TensorFlowLibraryVersion v : versionsModel.getFilteredVersions()) {
			final JRadioButton btn = new JRadioButton(v.toString());
			btn.setOpaque(false);
			// TODO tooltip?
			// TODO selected
			// TODO action
			versionGroup.add(btn);
		}
	}

	private Component createStatusPanel() {
		final JPanel statusPanel = new JPanel(new MigLayout());
		final JLabel status = new JLabel();
		status.setFont(status.getFont().deriveFont(Font.PLAIN));
		statusPanel.add(status);
		return statusPanel;
	}

	private JLabel createLabel(final String text) {
		JLabel label = new JLabel(text);
		label.setHorizontalAlignment(SwingConstants.RIGHT);
		label.setHorizontalTextPosition(SwingConstants.RIGHT);
		return label;
	}

	private void updateFilter() {
		// TODO remove and call update install choices directly?
		installPanel.removeAll();
		updateInstallChoices();
		installPanel.revalidate();
		installPanel.repaint();
	}
}
