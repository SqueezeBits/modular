# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import functools
import inspect
import json
import os
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, NoReturn


class ConfigDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        """Initialize ConfigDict."""
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs) -> NoReturn:
        """Set default value."""
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs) -> NoReturn:
        """Pop item."""
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs) -> NoReturn:
        """Update dictionary."""
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __setattr__(self, name: str, value: Any):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(
                f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
            )
        super().__setattr__(name, value)

    def __setitem__(self, name: str, value: Any):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(
                f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
            )
        super().__setitem__(name, value)


class ConfigMixin:
    config_name = None

    @classmethod
    def load_config(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> dict:
        """Load configuration from a pretrained model directory.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing the configuration.
        """
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        subfolder = kwargs.pop("subfolder", None)

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if subfolder is not None and os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path, subfolder, cls.config_name
                )
            ):
                config_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, cls.config_name
                )
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, cls.config_name)
            ):
                # Load from a pretrained checkpoint
                config_file = os.path.join(
                    pretrained_model_name_or_path, cls.config_name
                )
            else:
                raise OSError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            raise ValueError(
                f"The provided pretrained_model_name_or_path '{pretrained_model_name_or_path}'"
                " is neither a valid local path nor downloaded properly from Hugging Face Hub."
            )

        config_dict = cls._dict_from_json_file(config_file)
        return config_dict

    @property
    def config(self) -> ConfigDict:
        """Returns the config of the class as a dictionary.

        Returns:
            `Dict[str, Any]`: Config of the class.
        """
        return self._internal_dict

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike) -> dict:
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @staticmethod
    def _get_init_keys(input_class: Any) -> set:
        if hasattr(input_class, "components"):
            return set(input_class.components.keys())
        return set(
            dict(inspect.signature(input_class.__init__).parameters).keys()
        )

    @classmethod
    def extract_init_dict(cls, config_dict: dict) -> dict:
        """Extract init dictionary from config dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Dictionary containing the init parameters.
        """
        expected_keys = cls._get_init_keys(cls)

        init_dict = {
            k: config_dict[k] for k in config_dict if k in expected_keys
        }
        return init_dict

    def register_to_config(self, **kwargs) -> None:
        """Register arguments to the config.

        Args:
            **kwargs: Arguments to register.
        """
        if self.config_name is None:
            raise NotImplementedError(
                f"Make sure that {self.__class__} has defined a class name `config_name`"
            )
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            internal_dict = {**self._internal_dict, **kwargs}

        self._internal_dict = ConfigDict(internal_dict)


def register_to_config(init: Callable) -> Callable:
    """Register arguments to the config.

    Args:
        init: Initialization function of a class.

    Returns:
        Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
        automatically sent to `self.register_to_config`. To ignore a specific argument accepted by the init but that
        shouldn't be registered in the config, use the `ignore_for_config` class variable.

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self: Any, *args, **kwargs) -> None:
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {
            k: v for k, v in kwargs.items() if k.startswith("_")
        }
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_to_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default
            for i, (name, p) in enumerate(signature.parameters.items())
            if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys(), strict=False):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )

        # Take note of the parameters that were not present in the loaded config
        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            new_kwargs["_use_default_values"] = list(
                set(new_kwargs.keys()) - set(init_kwargs)
            )

        new_kwargs = {**config_init_kwargs, **new_kwargs}
        self.register_to_config(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init
