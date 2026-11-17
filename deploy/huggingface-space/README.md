---
title: Vietnamese ASR STT19
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
python_version: 3.12
preload_from_hub:
  - nguyenvulebinh/wav2vec2-base-vietnamese-250h vi_lm_4grams.bin.zip 69e9000591623e5a4fc2f502407860bcdc0de0b2
---

# Vietnamese AI Speech Recognition — STT19

Demo so sánh hai kiến trúc nhận dạng tiếng nói tiếng Việt:

- Wav2Vec2 CTC: `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, kết hợp language model 4-gram.
- PhoWhisper Base Seq2Seq: checkpoint thích nghi có giám sát cho mẫu ST23D.

Audio được xử lý trong Space, không gửi sang dịch vụ ASR bên ngoài. Kết quả không được cam kết chính xác 100%; WER/CER chỉ có ý nghĩa khi người dùng cung cấp transcript tham chiếu đúng.
