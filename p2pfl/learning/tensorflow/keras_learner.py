#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Keras learner for P2PFL."""

from typing import Dict, List, Tuple

import tensorflow as tf
from keras import callbacks

from p2pfl.learning.callbacks.tensorflow.keras_logger import FederatedLogger
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.framework_identifier import FrameworkIdentifier
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.tensorflow.keras_dataset import KerasExportStrategy
from p2pfl.learning.tensorflow.keras_model import KerasModel
from p2pfl.management.logger import logger


class KerasLearner(NodeLearner):
    """
    Learner for TensorFlow/Keras models in P2PFL.

    Args:
        model: The KerasModel instance.
        data: The P2PFLDataset instance.
        self_addr: The address of this node.

    """

    def __init__(
        self, model: KerasModel, data: P2PFLDataset, self_addr: str = "unknown-node", callbacks: List[callbacks.Callback] = None
    ) -> None:
        """Initialize the KerasLearner."""
        super().__init__(model, data, self_addr, callbacks)
        self.callbacks.append(FederatedLogger(self_addr))
        # Compile the model (you might need to customize this)
        self.model.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def __get_tf_model_data(self, train: bool = True) -> Tuple[tf.keras.Model, tf.data.Dataset]:
        # Get Model
        tf_model = self.model.get_model()
        if not isinstance(tf_model, tf.keras.Model):
            raise ValueError("The model must be a TensorFlow Keras model")
        # Get Data
        data = self.data.export(KerasExportStrategy, train=train)
        if not isinstance(data, tf.data.Dataset):
            raise ValueError("The data must be a TensorFlow Dataset")
        return tf_model, data

    def fit(self) -> P2PFLModel:
        """Fit the model."""
        try:
            if self.epochs > 0:
                model, data = self.__get_tf_model_data(train=True)
                self.set_callbacks_additional_info(self.callbacks)
                model.fit(
                    data,
                    epochs=self.epochs,
                    # callbacks=[FederatedLogger(self.__self_addr)],
                    callbacks=self.callbacks,
                )
                self.get_callbacks_additional_info(self.callbacks)
            # Set model contribution
            self.model.set_contribution([self._self_addr], self.data.get_num_samples(train=True))

            return self.model
        except Exception as e:
            logger.error(self._self_addr, f"Error in training with Keras: {e}")
            raise e

    def interrupt_fit(self) -> None:
        """Interrupt the training process."""
        # Keras doesn't have a direct way to interrupt fit.
        # Need to implement a custom callback or use a flag to stop training.
        logger.error(self._self_addr, "Interrupting training (not fully implemented for Keras).")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the Keras model."""
        try:
            if self.epochs > 0:
                model, data = self.__get_tf_model_data(train=False)
                results = model.evaluate(data, verbose=0)
                if not isinstance(results, list):
                    results = [results]
                results_dict = dict(zip(model.metrics_names, results))
                for k, v in results_dict.items():
                    logger.log_metric(self._self_addr, k, v)
                return results_dict
            else:
                return {}
        except Exception as e:
            logger.error(self._self_addr, f"Evaluation error with Keras: {e}")
            raise e

    @staticmethod
    def get_framework() -> str:
        """
        Retrieve the framework name used by the learner.

        Returns:
            str: The framework name ('tensorflow').

        """
        return FrameworkIdentifier.KERAS.value
