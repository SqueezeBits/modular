# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.weights import WeightData, Weights
from max.nn.layer import Module
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import traced

from .model_config import AutoencoderKLFlux2Config
from .vae import Decoder, DiagonalGaussianDistribution, Encoder


class AutoencoderKLFlux2(Module):
    def __init__(self, config: AutoencoderKLFlux2Config) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=tuple(config.down_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            double_z=True,
            mid_block_add_attention=config.mid_block_add_attention,
            use_quant_conv=config.use_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )
        self.decoder = Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            norm_type="group",
            mid_block_add_attention=config.mid_block_add_attention,
            use_post_quant_conv=config.use_post_quant_conv,
            device=config.device,
            dtype=config.dtype,
        )

    def __call__(
        self,
        z: TensorValue,
        temb: TensorValue | None = None,
    ) -> TensorValue:
        return self.decoder(z, temb)


class PostprocessAndDecode(Module):
    def __init__(
        self,
        decoder: Decoder,
        *,
        batch_norm_eps: float,
        num_channels: int,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.batch_norm_eps = batch_norm_eps
        self._num_channels = num_channels
        self._device = device
        self._dtype = dtype
        self.bn_mean = Weight(
            name="bn_mean",
            dtype=dtype,
            shape=(num_channels,),
            device=device,
        )
        self.bn_var = Weight(
            name="bn_var",
            dtype=dtype,
            shape=(num_channels,),
            device=device,
        )

    def __call__(
        self,
        latents_bsc: TensorValue,
        h_carrier: TensorValue,
        w_carrier: TensorValue,
    ) -> TensorValue:
        batch = latents_bsc.shape[0]
        c = latents_bsc.shape[2]
        h = h_carrier.shape[0]
        w = w_carrier.shape[0]

        latents_bsc = ops.rebind(latents_bsc, [batch, h * w, c])
        latents_bhwc = ops.reshape(latents_bsc, (batch, h, w, c))
        latents = ops.permute(latents_bhwc, [0, 3, 1, 2])

        bn_mean = ops.reshape(self.bn_mean.cast(latents.dtype), (1, c, 1, 1))
        bn_var = ops.reshape(self.bn_var.cast(latents.dtype), (1, c, 1, 1))
        latents = latents * ops.sqrt(bn_var + self.batch_norm_eps) + bn_mean

        latents = ops.reshape(latents, (batch, c // 4, 2, 2, h, w))
        latents = ops.permute(latents, [0, 1, 4, 2, 5, 3])
        latents = ops.reshape(latents, (batch, c // 4, h * 2, w * 2))

        decoded = self.decoder(latents, None)
        decoded = ops.permute(decoded, [0, 2, 3, 1])
        decoded = ops.min(ops.max(decoded * 0.5 + 0.5, 0.0), 1.0)
        return ops.transfer_to(
            ops.cast(decoded * 255.0, DType.uint8),
            DeviceRef.CPU(),
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", "seq", self._num_channels],
                device=self._device,
            ),
            TensorType(
                DType.float32, shape=["latent_h"], device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=["latent_w"], device=DeviceRef.CPU()
            ),
        )


class AutoencoderKLFlux2Model(ComponentModel):
    bn_running_mean: Buffer
    bn_running_var: Buffer

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.session = session
        self.config = AutoencoderKLFlux2Config.generate(
            config,
            encoding,
            devices,
        )
        self.encoder_model: Callable[[Buffer], tuple[Buffer, Buffer]] | None = (
            None
        )
        self._fused_model: Callable[[Buffer, Buffer, Buffer], Buffer] | None = (
            None
        )
        self.load_model()

    @staticmethod
    def _compile_module(
        *,
        session: InferenceSession,
        graph_name: str,
        module: Module,
        input_types: tuple[TensorType, ...],
    ) -> Model:
        with Graph(graph_name, input_types=input_types) as graph:
            outputs = module(*(value.tensor for value in graph.inputs))
            if isinstance(outputs, tuple):
                graph.output(*outputs)
            else:
                graph.output(outputs)
        return session.load(graph, weights_registry=module.state_dict())

    @staticmethod
    def _convert_weight_dtype(
        weight_data: WeightData,
        target_dtype: DType,
    ) -> WeightData:
        if (
            weight_data.dtype != target_dtype
            and weight_data.dtype.is_float()
            and target_dtype.is_float()
        ):
            return weight_data.astype(target_dtype)
        return weight_data

    @traced(message="AutoencoderKLFlux2Model.load_model")
    def load_model(self) -> Callable[..., Any]:
        encoder_state_dict: dict[str, WeightData] = {}
        bn_mean_data: WeightData | None = None
        bn_var_data: WeightData | None = None
        target_dtype = self.config.dtype

        for key, value in self.weights.items():
            weight_data = self._convert_weight_dtype(value.data(), target_dtype)
            if key == "bn.running_mean":
                bn_mean_data = weight_data
            elif key == "bn.running_var":
                bn_var_data = weight_data
            elif key.startswith("encoder."):
                encoder_state_dict[key.removeprefix("encoder.")] = weight_data
            elif key.startswith("quant_conv."):
                encoder_state_dict[key] = weight_data

        if bn_mean_data is None or bn_var_data is None:
            raise ValueError(
                "BatchNorm statistics (running_mean, running_var) not loaded. "
                "Make sure the model weights contain 'bn.running_mean' and 'bn.running_var'."
            )

        self.bn_running_mean = Buffer.from_dlpack(bn_mean_data.data).to(
            self.devices[0]
        )
        self.bn_running_var = Buffer.from_dlpack(bn_var_data.data).to(
            self.devices[0]
        )

        autoencoder = AutoencoderKLFlux2(self.config)
        autoencoder.encoder.load_state_dict(
            encoder_state_dict,
            weight_alignment=1,
            strict=True,
        )
        with Graph(
            "flux2_autoencoder_encoder",
            input_types=autoencoder.encoder.input_types(),
        ) as graph:
            moments = autoencoder.encoder(
                *(value.tensor for value in graph.inputs)
            )
            mean, logvar = ops.split(
                moments,
                [self.config.latent_channels, self.config.latent_channels],
                axis=1,
            )
            graph.output(mean, logvar)
        encoder_model = self.session.load(
            graph,
            weights_registry=autoencoder.encoder.state_dict(),
        )

        def execute_encoder(sample: Buffer) -> tuple[Buffer, Buffer]:
            outputs = encoder_model.execute(sample)
            if len(outputs) != 2:
                raise RuntimeError(
                    f"Expected encoder to return 2 tensors, got {len(outputs)}."
                )
            return outputs[0], outputs[1]

        self.encoder_model = execute_encoder
        return self.encoder_model

    @traced(message="AutoencoderKLFlux2Model.build_fused_decode")
    def build_fused_decode(
        self, device: Device, num_channels: int
    ) -> Callable[..., Any]:
        if self._fused_model is not None:
            return self._fused_model

        fused_weights: dict[str, WeightData] = {}
        target_dtype = self.config.dtype
        for key, value in self.weights.items():
            weight_data = self._convert_weight_dtype(value.data(), target_dtype)
            if key.startswith("decoder."):
                fused_weights[key] = weight_data
            elif key.startswith("post_quant_conv."):
                fused_weights[f"decoder.{key}"] = weight_data
            elif key == "bn.running_mean":
                fused_weights["bn_mean"] = weight_data
            elif key == "bn.running_var":
                fused_weights["bn_var"] = weight_data

        autoencoder = AutoencoderKLFlux2(self.config)
        fused = PostprocessAndDecode(
            decoder=autoencoder.decoder,
            batch_norm_eps=self.config.batch_norm_eps,
            num_channels=num_channels,
            device=DeviceRef.from_device(device),
            dtype=target_dtype,
        )
        fused.load_state_dict(
            fused_weights,
            weight_alignment=1,
            strict=True,
        )
        fused_model = self._compile_module(
            session=self.session,
            graph_name="flux2_autoencoder_fused_decode",
            module=fused,
            input_types=fused.input_types(),
        )

        def execute_fused_decode(
            latents_bsc: Buffer,
            h_carrier: Buffer,
            w_carrier: Buffer,
        ) -> Buffer:
            outputs = fused_model.execute(latents_bsc, h_carrier, w_carrier)
            if len(outputs) != 1:
                raise RuntimeError(
                    f"Expected fused decode to return 1 tensor, got {len(outputs)}."
                )
            return outputs[0]

        self._fused_model = execute_fused_decode
        return self._fused_model

    def encode(
        self, sample: Buffer, return_dict: bool = True
    ) -> dict[str, DiagonalGaussianDistribution] | DiagonalGaussianDistribution:
        if self.encoder_model is None:
            raise ValueError(
                "Encoder not loaded. Check if encoder weights exist in the model."
            )

        mean, logvar = self.encoder_model(sample)
        posterior = DiagonalGaussianDistribution(mean=mean, logvar=logvar)
        if return_dict:
            return {"latent_dist": posterior}
        return posterior

    @property
    def bn(self) -> SimpleNamespace:
        return SimpleNamespace(
            running_mean=self.bn_running_mean,
            running_var=self.bn_running_var,
        )
