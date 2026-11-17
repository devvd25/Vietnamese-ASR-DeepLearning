# Hướng dẫn demo Gradio

## Chạy

```powershell
.venv\Scripts\Activate.ps1
python scripts/launch_demo.py --port 7860
```

Truy cập `http://127.0.0.1:7860`.

## Luồng sử dụng

1. Tải WAV/FLAC/OGG/MP3/M4A hoặc ghi âm.
2. Nghe lại audio và kiểm tra thông tin file.
3. Chọn **Wav2Vec2 — Vietnamese 250h** hoặc **PhoWhisper Base — ST23D supervised adaptation**;
   Whisper chỉ hiện khi checkpoint local hợp lệ.
4. Có thể nhập transcript chuẩn để tính metric cho đúng mẫu đó.
5. Nhấn **Nhận dạng tiếng nói** và chờ progress.

## Guardrails

- Demo không giới hạn tổng thời lượng; audio dài được chia thành các cửa sổ 28 giây và ghép transcript theo thứ tự.
- Audio rỗng, sai định dạng, không đọc được hoặc vượt 100 MB trả thông báo tiếng Việt.
- Model được cache, không nạp lại mỗi lần nhấn.
- Không gọi paid API và không gửi audio tới dịch vụ ASR ngoài.
- Hiển thị proxy độ tin cậy từ xác suất frame/token và ghi rõ chỉ số chưa hiệu
  chuẩn, không phải xác suất toàn câu đúng.
- Không tính WER/CER nếu không có reference.

## LAN hoặc public share

LAN: `python scripts/launch_demo.py --server-name 0.0.0.0`. Chỉ dùng `--share` khi hiểu rằng Gradio sẽ tạo tunnel; mặc định dự án chỉ lắng nghe localhost để bảo vệ audio.
