
from max.driver import DeviceSpec, Device, CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.tensor import Tensor

from max.pipelines.architectures.flux2.model import Flux2Transformer2DModel
from max.pipelines.architectures.flux2.model_config import Flux2ConfigBase
import numpy as np

def verify_custom_op():
    print("Verifying custom op integration...")
    
    # 1. Setup Config
    # Use smaller config to execute fast
    config = Flux2ConfigBase(
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=1, # Minimal layers
        num_single_layers=1, # Minimal layers
        attention_head_dim=128,  # Must match rope_dim (sum of axes_dim = 32*4 = 128)
        num_attention_heads=4,
        joint_attention_dim=128,
        timestep_guidance_channels=256,
        mlp_ratio=4.0,
        dtype=DType.float32,
        device=DeviceRef.GPU(),
    )
    
    # 2. Instantiate Model
    print("Instantiating Flux2Transformer2DModel...")
    model = Flux2Transformer2DModel(config)
    
    # 3. Create dummy inputs
    batch_size = 1
    image_seq_len = 16 # Small seq len
    text_seq_len = 8
    
    hidden_states = Tensor.zeros([batch_size, image_seq_len, config.in_channels], dtype=config.dtype, device=config.device.to_device())
    encoder_hidden_states = Tensor.zeros([batch_size, text_seq_len, config.joint_attention_dim], dtype=config.dtype, device=config.device.to_device())
    timestep = Tensor.zeros([batch_size], dtype=config.dtype, device=config.device.to_device())
    img_ids = Tensor.zeros([batch_size, image_seq_len, 4], dtype=DType.int64, device=config.device.to_device())
    txt_ids = Tensor.zeros([batch_size, text_seq_len, 4], dtype=DType.int64, device=config.device.to_device())
    guidance = Tensor.zeros([batch_size], dtype=config.dtype, device=config.device.to_device())
    
    # 4. Compile with custom extensions
    print("Compiling model (this should trigger custom op loading)...")
    input_types = model.input_types_with_shapes(batch_size, image_seq_len, text_seq_len)
    
    # Pass dummy weights (random)
    state_dict = {}
    for name, param in model.parameters:
        state_dict[name] = Tensor.zeros(param.shape, dtype=param.dtype, device=CPU())


    try:
        compiled_model = model.compile(
            *input_types,
            weights=state_dict,
        )
        print("Compilation successful!")
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Run model
    print("Running model...")
    try:
        output = compiled_model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance
        )
        print("Run successful!")
        print(f"Output shape: {output[0].shape}")
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_custom_op()
