import sys
import os
import torch
from types import SimpleNamespace

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))

from tools.inference_utils import FinCast_Inference

def test_load():
    print("Initializing FinCast Model...")
    
    config = SimpleNamespace()
    config.backend = "cpu"
    config.model_path = "fincast_model/v1.pth"
    config.model_version = "v1"
    
    # Minimal config for initialization
    config.data_path = "data/BTC_USDT_6h.csv"
    config.data_frequency = "6h"
    config.context_len = 128
    config.horizon_len = 32
    config.all_data = False
    config.columns_target = ['close']
    config.series_norm = True
    config.batch_size = 1
    config.forecast_mode = "mean"
    config.save_output = False
    config.plt_outputs = False

    try:
        print(f"Loading weights from {config.model_path} (This may take a minute)...")
        # The constructor of FinCast_Inference handles model loading
        fincast = FinCast_Inference(config)
        print("✅ FinCast Model successfully initialized and weights loaded on CPU!")
        
        # Check model size in memory
        param_size = 0
        for param in fincast.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in fincast.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"Model size in RAM: {size_all_mb:.2f} MB")
        
    except Exception as e:
        print(f"❌ Failed to load FinCast: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
