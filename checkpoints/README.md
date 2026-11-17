# Checkpoints

Thư mục này không lưu model weights trong Git. `scripts/train_model.py` lưu checkpoint theo `config.yaml`, sao chép best model hợp lệ sang `checkpoints/<experiment>/best`, và có thể resume bằng `--resume-from-checkpoint`.

Một thư mục checkpoint hợp lệ phải có ít nhất `config.json`, model weights (`model.safetensors` hoặc `pytorch_model.bin`) và các file processor/tokenizer.
