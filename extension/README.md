# Vietnamese ASR Live Captions

Chrome Extension Manifest V3 tạo phụ đề tiếng Việt gần realtime cho video YouTube bằng hai model local của đồ án:

- `Wav2Vec2 — Vietnamese 250h`: mặc định, tốc độ nhanh hơn.
- `PhoWhisper Base — VietSpeech 10h`: model fine-tuned để so sánh.

## 1. Chạy server local

Từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\launch_realtime_server.py --config config.phowhisper_vietspeech_10h.yaml --port 8765
```

Kiểm tra server tại `http://127.0.0.1:8765/health`.

## 2. Cài extension dạng unpacked

1. Mở `chrome://extensions` hoặc `edge://extensions`.
2. Bật **Developer mode**.
3. Chọn **Load unpacked**.
4. Chọn đúng thư mục `extension` của dự án.

Chrome/Edge 116 trở lên là bắt buộc vì extension dùng stream ID của `tabCapture` trong offscreen document.

## 3. Sử dụng

1. Mở video trên `https://www.youtube.com` và phát video.
2. Bấm biểu tượng extension.
3. Chọn model và cửa sổ nhận dạng 3–6 giây.
4. Bấm **Bắt đầu phụ đề**.
5. Bấm **Dừng** khi hoàn tất.

Đoạn dài hơn cho model nhiều ngữ cảnh hơn nhưng tăng độ trễ. Lần dùng model đầu tiên chậm hơn do phải nạp weights vào GPU/RAM.

## 4. Quyền và riêng tư

- `tabCapture`: chỉ thu âm thanh của tab sau thao tác chủ động của người dùng.
- `offscreen`: giữ pipeline Web Audio hoạt động khi popup đóng.
- `activeTab`: giới hạn thao tác vào tab mà người dùng đang mở.
- `storage`: lưu trạng thái và model được chọn trong phiên trình duyệt.

PCM chỉ được gửi tới WebSocket loopback `ws://127.0.0.1:8765`. Server từ chối origin không có dạng `chrome-extension://…`; không gọi dịch vụ ASR bên ngoài và không tải video YouTube về đĩa.

## 5. Giới hạn kỹ thuật

- Đây là **near realtime**, không phải tức thời tuyệt đối; độ trễ thường từ 3–6 giây cộng thời gian suy luận.
- Câu nằm sát ranh giới giữa hai đoạn có thể thiếu từ hoặc lặp từ.
- Chất lượng phụ thuộc giọng nói, nhạc nền, tiếng ồn và miền nội dung.
- Laptop và server Python local phải tiếp tục chạy.
- Không thể bảo đảm chính xác 100% trên video mới.

## 6. Kiểm thử

```powershell
python -m pytest tests\test_realtime_caption.py tests\test_realtime_server.py tests\test_realtime_cli.py tests\test_youtube_extension_package.py -q
node --test tests\js\service_worker.test.mjs
python tests\browser_extension_smoke.py
```

