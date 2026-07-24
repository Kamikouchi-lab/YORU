# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Base class for training engine plugins.

This module is part of YORU core and is NOT subject to any plugin's license.
"""


class TrainerBase:
    """Abstract interface that all training plugins must implement."""

    def train(self, config: dict):
        """Start training with the given configuration.

        config keys:
            img_size (int): image size
            batch_size (int): batch size
            epochs (int): number of epochs
            data_yaml (str): path to data YAML file
            weights (str): path to pretrained weights
            project_dir (str): output project directory

        Returns:
            subprocess.Popen: the training process handle.
        """
        raise NotImplementedError
