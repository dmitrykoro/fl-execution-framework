"""
Unit tests for text classification model with LoRA.

Tests text classifier model loading, LoRA configuration, state management,
and parameter extraction with lightweight mock implementations.
"""

from collections import OrderedDict
from unittest.mock import Mock, patch

import torch

from src.network_models.text_classifier_model import (
    get_lora_state_dict,
    load_text_classifier_with_lora,
    load_text_classifier_without_lora,
    set_lora_state_dict,
)
from tests.common import np


class TestTextClassifierLoading:
    """Test text classifier model loading functions."""

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    def test_load_text_classifier_with_lora_default_params(
        self, mock_get_peft_model, mock_auto_model
    ) -> None:
        """Test loading text classifier with default LoRA parameters."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        model = load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased", num_labels=2
        )
        mock_get_peft_model.assert_called_once()
        assert model == mock_lora_model

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    @patch("src.network_models.text_classifier_model.LoraConfig")
    def test_load_text_classifier_with_lora_custom_params(
        self, mock_lora_config, mock_get_peft_model, mock_auto_model
    ) -> None:
        """Test loading text classifier with custom LoRA parameters."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        mock_config = Mock()
        mock_lora_config.return_value = mock_config

        model = load_text_classifier_with_lora(
            model_name="bert-base-uncased",
            num_labels=5,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.2,
            lora_target_modules=["query", "key", "value"],
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "bert-base-uncased", num_labels=5
        )
        mock_lora_config.assert_called_once_with(
            r=16,
            lora_alpha=32,
            lora_dropout=0.2,
            target_modules=["query", "key", "value"],
            task_type="SEQ_CLS",
        )
        mock_get_peft_model.assert_called_once_with(mock_base_model, mock_config)
        assert model == mock_lora_model

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    def test_load_text_classifier_with_lora_multiclass(
        self, mock_get_peft_model, mock_auto_model
    ) -> None:
        """Test loading text classifier for multiclass classification."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        model = load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=10,
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased", num_labels=10
        )
        assert model == mock_lora_model

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    def test_load_text_classifier_without_lora(self, mock_auto_model) -> None:
        """Test loading text classifier without LoRA."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        model = load_text_classifier_without_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased", num_labels=2
        )
        assert model == mock_model

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    def test_load_text_classifier_without_lora_multiclass(
        self, mock_auto_model
    ) -> None:
        """Test loading text classifier without LoRA for multiclass."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        model = load_text_classifier_without_lora(
            model_name="bert-base-uncased",
            num_labels=5,
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "bert-base-uncased", num_labels=5
        )
        assert model == mock_model


class TestLoraStateDictFunctions:
    """Test LoRA state dict operations for text classifier."""

    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    def test_get_lora_state_dict(self, mock_get_peft_state_dict) -> None:
        """Test getting LoRA state dict as numpy arrays."""
        mock_state_dict = OrderedDict(
            {
                "base_model.model.distilbert.transformer.layer.0.attention.q_lin.lora_A.default.weight": torch.randn(
                    8, 768
                ),
                "base_model.model.distilbert.transformer.layer.0.attention.q_lin.lora_B.default.weight": torch.randn(
                    768, 8
                ),
            }
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        result = get_lora_state_dict(mock_model)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)
        mock_get_peft_state_dict.assert_called_once_with(mock_model)

    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    def test_get_lora_state_dict_empty(self, mock_get_peft_state_dict) -> None:
        """Test getting LoRA state dict when empty."""
        mock_state_dict = OrderedDict()
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        result = get_lora_state_dict(mock_model)

        assert isinstance(result, list)
        assert len(result) == 0
        mock_get_peft_state_dict.assert_called_once_with(mock_model)

    @patch("src.network_models.text_classifier_model.set_peft_model_state_dict")
    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    def test_set_lora_state_dict(
        self, mock_get_peft_state_dict, mock_set_peft_state_dict
    ) -> None:
        """Test setting LoRA state dict from numpy arrays."""
        mock_state_dict = OrderedDict(
            {
                "lora_A.weight": torch.randn(8, 768),
                "lora_B.weight": torch.randn(768, 8),
            }
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        state_list = [np.random.randn(8, 768), np.random.randn(768, 8)]

        set_lora_state_dict(mock_model, state_list)

        mock_get_peft_state_dict.assert_called_once_with(mock_model)
        mock_set_peft_state_dict.assert_called_once()

        call_args = mock_set_peft_state_dict.call_args
        assert call_args[0][0] == mock_model
        assert isinstance(call_args[0][1], OrderedDict)

    @patch("src.network_models.text_classifier_model.set_peft_model_state_dict")
    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    def test_set_lora_state_dict_tensor_conversion(
        self, mock_get_peft_state_dict, mock_set_peft_state_dict
    ) -> None:
        """Test that numpy arrays are converted to tensors."""
        mock_state_dict = OrderedDict(
            {
                "lora_A.weight": torch.randn(4, 256),
            }
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        state_list = [np.random.randn(4, 256)]

        set_lora_state_dict(mock_model, state_list)

        call_args = mock_set_peft_state_dict.call_args
        loaded_state_dict = call_args[0][1]

        # Verify all values are tensors
        for value in loaded_state_dict.values():
            assert isinstance(value, torch.Tensor)

    def test_lora_state_dict_round_trip(self) -> None:
        """Test that LoRA state dict can be extracted and restored."""
        mock_model = Mock()

        with (
            patch(
                "src.network_models.text_classifier_model.get_peft_model_state_dict"
            ) as mock_get,
            patch(
                "src.network_models.text_classifier_model.set_peft_model_state_dict"
            ) as mock_set,
        ):
            original_state_dict = OrderedDict(
                {
                    "lora_A": torch.randn(8, 768),
                    "lora_B": torch.randn(768, 8),
                }
            )
            mock_get.return_value = original_state_dict

            # Extract state
            state_list = get_lora_state_dict(mock_model)

            # Restore state
            set_lora_state_dict(mock_model, state_list)

            assert mock_get.call_count == 2
            mock_set.assert_called_once()


class TestLoraConfiguration:
    """Test LoRA configuration settings."""

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    @patch("src.network_models.text_classifier_model.LoraConfig")
    def test_lora_config_task_type(
        self, mock_lora_config, mock_get_peft_model, mock_auto_model
    ) -> None:
        """Test that LoRA config sets correct task type for sequence classification."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        mock_config = Mock()
        mock_lora_config.return_value = mock_config

        load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
        )

        # Verify task_type is SEQ_CLS
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["task_type"] == "SEQ_CLS"

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    @patch("src.network_models.text_classifier_model.LoraConfig")
    def test_lora_config_default_target_modules(
        self, mock_lora_config, mock_get_peft_model, mock_auto_model
    ) -> None:
        """Test that default target modules are q_lin and v_lin for DistilBERT."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        mock_config = Mock()
        mock_lora_config.return_value = mock_config

        load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
        )

        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["target_modules"] == ["q_lin", "v_lin"]

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    def test_lora_suppresses_trainable_params_output(
        self, mock_get_peft_model, mock_auto_model
    ) -> None:
        """Test that print_trainable_parameters output is suppressed."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        # Should not raise or print anything
        load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
        )

        mock_lora_model.print_trainable_parameters.assert_called_once()


class TestModelParameterShapes:
    """Test that model parameters have expected shapes."""

    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    def test_lora_state_dict_preserves_shapes(self, mock_get_peft_state_dict) -> None:
        """Test that state dict extraction preserves tensor shapes."""
        original_tensors = {
            "lora_A": torch.randn(8, 768),
            "lora_B": torch.randn(768, 8),
            "lora_C": torch.randn(16, 512),
        }
        mock_state_dict = OrderedDict(original_tensors)
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        result = get_lora_state_dict(mock_model)

        # Check shapes are preserved
        for i, (key, tensor) in enumerate(original_tensors.items()):
            assert result[i].shape == tuple(tensor.shape)

    @patch("src.network_models.text_classifier_model.set_peft_model_state_dict")
    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    def test_set_lora_state_dict_preserves_shapes(
        self, mock_get_peft_state_dict, mock_set_peft_state_dict
    ) -> None:
        """Test that setting state dict preserves shapes."""
        mock_state_dict = OrderedDict(
            {
                "lora_A": torch.randn(8, 768),
                "lora_B": torch.randn(768, 8),
            }
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        state_list = [
            np.random.randn(8, 768),
            np.random.randn(768, 8),
        ]

        set_lora_state_dict(mock_model, state_list)

        call_args = mock_set_peft_state_dict.call_args
        loaded_state_dict = call_args[0][1]

        # Verify shapes match
        assert loaded_state_dict["lora_A"].shape == torch.Size([8, 768])
        assert loaded_state_dict["lora_B"].shape == torch.Size([768, 8])


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    def test_load_with_single_label(self, mock_get_peft_model, mock_auto_model) -> None:
        """Test loading model with single label (regression-like task)."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        model = load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=1,
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased", num_labels=1
        )
        assert model == mock_lora_model

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    def test_load_with_zero_dropout(self, mock_get_peft_model, mock_auto_model) -> None:
        """Test loading model with zero dropout."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
            lora_dropout=0.0,
        )

        mock_get_peft_model.assert_called_once()

    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    def test_get_lora_state_dict_with_large_tensors(
        self, mock_get_peft_state_dict
    ) -> None:
        """Test state dict extraction with large tensors."""
        mock_state_dict = OrderedDict(
            {
                "large_tensor": torch.randn(1024, 4096),
            }
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        result = get_lora_state_dict(mock_model)

        assert len(result) == 1
        assert result[0].shape == (1024, 4096)


class TestIntegration:
    """Integration tests for text classifier model functions."""

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    @patch("src.network_models.text_classifier_model.get_peft_model")
    @patch("src.network_models.text_classifier_model.get_peft_model_state_dict")
    @patch("src.network_models.text_classifier_model.set_peft_model_state_dict")
    def test_full_workflow_with_lora(
        self,
        mock_set_peft_state_dict,
        mock_get_peft_state_dict,
        mock_get_peft_model,
        mock_auto_model,
    ) -> None:
        """Test complete workflow: load model, extract state, modify, restore."""
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        # Load model
        model = load_text_classifier_with_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
        )

        # Extract state
        mock_state_dict = OrderedDict(
            {
                "lora_A": torch.randn(8, 768),
                "lora_B": torch.randn(768, 8),
            }
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        state_list = get_lora_state_dict(model)

        # Modify state (simulate aggregation)
        modified_state = [arr * 0.5 for arr in state_list]

        # Restore state
        set_lora_state_dict(model, modified_state)

        assert mock_auto_model.from_pretrained.called
        assert mock_get_peft_model.called
        assert mock_get_peft_state_dict.call_count == 2
        assert mock_set_peft_state_dict.called

    @patch(
        "src.network_models.text_classifier_model.AutoModelForSequenceClassification"
    )
    def test_full_workflow_without_lora(self, mock_auto_model) -> None:
        """Test workflow without LoRA (full fine-tuning)."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        # Load model
        model = load_text_classifier_without_lora(
            model_name="distilbert-base-uncased",
            num_labels=2,
        )

        mock_auto_model.from_pretrained.assert_called_once()
        assert model == mock_model
