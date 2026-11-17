# Kế Hoạch Tái Cấu Trúc Giao Diện (UI/UX Revamp Plan)
**Dự án:** Vietnamese AI Speech Recognition (Deep Learning STT 19)  
**Tiêu chuẩn áp dụng:** `/ui-ux-pro-max` (Minimalism & Swiss Technical Workbench)  
**Tôn chỉ cốt lõi:** Màu sắc gọn gàng, tương phản cao, chuyên nghiệp, **tuyệt đối không dùng icon / emoji bừa bãi**.

---

## 1. Tôn Chỉ Thiết Kế (Design Manifesto)

1. **Zero Emoji Clutter (Loại bỏ triệt để biểu tượng bừa bãi):**
   - Tuyệt đối không dùng các emoji hoạt họa (như 🎙️, 🚀, 🤖, ✨, 📊, ⚡, ❌, ✅) trong giao diện. Emoji không đồng nhất trên các hệ điều hành (Windows/macOS/Linux), tạo cảm giác nghiệp dư và làm xao nhãng việc đọc kết quả.
   - Thay thế bằng:
     - **Status Indicator Dots:** Chấm trạng thái nhỏ gọn (`●`) với màu sắc ngữ nghĩa rõ ràng.
     - **Technical Badges:** Nhãn văn bản viết hoa bọc khung mảnh (ví dụ: `[ON-DEVICE]`, `[LOCAL]`, `[CUDA]`).
     - **Minimalist SVG (nếu cần):** Các nét vẽ vector mảnh (stroke 1.5px), cùng tông màu với hệ thống.

2. **Màu Sắc Gọn Gàng & Khoa Học (Disciplined Technical Palette):**
   - Loại bỏ nền trắng chói (`#FFFFFF` thuần trên toàn màn hình). Sử dụng nền xám đá phiến (`Slate 50 - #F8FAFC`) để làm nổi bật các khối chức năng thẻ (Card Surfaces), bảo vệ mắt người dùng khi làm việc lâu.
   - Độ tương phản chữ/nền tối thiểu 7:1 (vượt tiêu chuẩn WCAG AAA), đảm bảo đọc rõ ràng các dấu thanh điệu tiếng Việt phức tạp.

3. **Cấu Trúc Bảng Chỉ Số Thông Minh (Structured Metric Grid):**
   - Chuyển đổi khối văn bản chạy thô sơ ("Thông tin chạy") thành bảng lưới 4 chỉ số kỹ thuật trực quan (Latency, RTF, Proxy Confidence, WER/CER).
   - Số liệu định dạng `font-variant-numeric: tabular-nums` giúp các con số luôn thẳng cột và dễ so sánh.

---

## 2. Bảng Màu & Design Tokens

### 2.1. Bảng màu hệ thống (Color Palette)

| Token | Mã màu Hex | Tên màu | Mục đích sử dụng |
| :--- | :--- | :--- | :--- |
| `--color-bg` | `#F8FAFC` | Slate 50 | Màu nền chính toàn trang (êm dịu, không lóa) |
| `--color-surface` | `#FFFFFF` | Pure White | Nền của các khối Card, Input, Textbox |
| `--color-border` | `#E2E8F0` | Slate 200 | Viền khung thẻ, phân cách nội dung sắc nét |
| `--color-border-subtle` | `#F1F5F9` | Slate 100 | Đường kẻ ngăn cách phụ bên trong thẻ |
| `--color-text-primary` | `#0F172A` | Slate 900 | Tiêu đề, chữ chính, độ sắc nét cao |
| `--color-text-secondary`| `#475569` | Slate 600 | Nhãn input, mô tả phụ, hướng dẫn |
| `--color-text-muted` | `#64748B` | Slate 500 | Chú thích nhỏ, metadata, disclaimer |
| `--color-brand` | `#2563EB` | Cobalt Blue 600 | Nút hành động chính, viền focus |
| `--color-brand-hover` | `#1D4ED8` | Cobalt Blue 700 | Trạng thái hover của nút bấm |
| `--color-success` | `#059669` | Emerald 600 | Trạng thái Local Ready, WER 0.00% |
| `--color-success-bg` | `#ECFDF5` | Emerald 50 | Nền badge trạng thái an toàn / sẵn sàng |
| `--color-danger` | `#DC2626` | Rose 600 | Trạng thái lỗi, file không hợp lệ |
| `--color-danger-bg` | `#FEF2F2` | Rose 50 | Nền thông báo lỗi |

### 2.2. Quy cách Không gian & Kiểu chữ (Spacing & Typography)

* **Font chữ:** `Inter`, `Be Vietnam Pro`, `-apple-system`, `system-ui`, sans-serif.
* **Thang khoảng cách (8pt Grid):** 4px, 8px, 12px, 16px, 20px, 24px, 32px.
* **Bo góc (Border Radius):**
  * Thẻ chính (Main Cards): `12px`
  * Nút bấm & Ô nhập liệu: `8px`
  * Status Badges / Pills: `9999px` (Full rounded)
* **Đổ bóng (Micro Elevation):** `box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.06), 0 1px 2px -1px rgba(15, 23, 42, 0.04)`.

---

## 3. Bố Cục Giao Diện Mới (ASCII Wireframe)

### 3.1. Phiên bản Desktop (Độ rộng khuyến nghị: 1120px)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  VIETNAMESE ASR BENCHMARK                      [● ON-DEVICE ENGINE] [● SYSTEM READY]   │
│  Bộ suy luận nhận dạng tiếng nói tự động cục bộ • Wav2Vec2 CTC & PhoWhisper Seq2Seq   │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ 1. DỮ LIỆU ĐẦU VÀO (AUDIO INPUT)          │ 2. CẤU HÌNH & THỰC THI (ENGINE CONTROLS)   │
│                                           │                                            │
│ ┌───────────────────────────────────────┐ │ Model suy luận:                            │
│ │ [Audio Dropzone / Microphone Player]  │ │ ┌────────────────────────────────────────┐ │
│ │ Kéo thả file WAV, FLAC, MP3 (Max 100M)│ │ │ PhoWhisper Base — Adaptation (ST23D)   ▼│ │
│ └───────────────────────────────────────┘ │ └────────────────────────────────────────┘ │
│ File info: audio_sample.wav • 8.52s • 16k │                                            │
│                                           │ Trạng thái checkpoint:                     │
│ Transcript tham chiếu (không bắt buộc):   │ ┌────────────────────────────────────────┐ │
│ ┌───────────────────────────────────────┐ │ │ ● Sẵn sàng suy luận (Mẫu thích nghi)   │ │
│ │ Nhập nội dung chuẩn nếu muốn tính WER/│ │ └────────────────────────────────────────┘ │
│ │ CER cho mẫu này...                    │ │                                            │
│ └───────────────────────────────────────┘ │ ┌────────────────────────────────────────┐ │
│                                           │ │         NHẬN DẠNG TIẾNG NÓI            │ │
│                                           │ └────────────────────────────────────────┘ │
│                                           │ Giới hạn: Chunk tối đa 28s • Max 100MB     │
├───────────────────────────────────────────┴────────────────────────────────────────────┤
│ 3. VĂN BẢN NHẬN DẠNG (TRANSCRIPTION OUTPUT)                                            │
│ ┌─────────────────────────────────────────────────────────────────────────┬──────────┐ │
│ │ xin chào chào mừng đến với lớp st23d mình là đặng công vũ xin chào tất   │ [SAO     │ │
│ │ cả các bạn nha                                                          │  CHÉP]   │ │
│ └─────────────────────────────────────────────────────────────────────────┴──────────┘ │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. BẢNG CHỈ SỐ KỸ THUẬT (PERFORMANCE METRICS)                                          │
│ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────────┐ │
│ │ THỜI GIAN SUY LUẬN│ │ ĐỘ TIN CẬY ƯỚC TÍNH│ │ WER (TỪ LỖI)     │ │ CER (KÝ TỰ LỖI)     │ │
│ │     0.342s       │ │      94.8%       │ │      0.00%       │ │        0.00%        │ │
│ │   (RTF: 0.04x)   │ │  Proxy CTC Non-Bl│ │  (Chuẩn tham chiếu│ │   (Chuẩn tham chiếu)│ │
│ └──────────────────┘ └──────────────────┘ └──────────────────┘ └─────────────────────┘ │
│ Thiết bị: cuda:0 • Kiến trúc: vinai/PhoWhisper-base • Phân đoạn: 1 đoạn                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Lưu ý kỹ thuật: Độ tin cậy hiển thị là proxy chưa hiệu chuẩn để tham khảo.             │
│ Mọi tác vụ suy luận chạy 100% cục bộ, không gửi audio tới máy chủ bên ngoài.          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Chi Tiết So Sánh: Trước vs. Sau Cải Tạo

| Thành phần | Hiện trạng (Before) | Đề xuất mới (After) |
| :--- | :--- | :--- |
| **Header Bar** | Văn bản trơn `DEEP LEARNING · STT 19` và tiêu đề H1 phẳng. | Top bar phân cấp rõ ràng với Status Pill: `[● ON-DEVICE]` (xanh) và phụ đề kỹ thuật súc tích. |
| **Tiêu đề phân đoạn** | `1. Chọn audio`, `2. Cấu hình`, `3. Kết quả` kiểu form thô. | Chuyển thành Section Headers chuyên nghiệp: `1. Dữ liệu âm thanh`, `2. Cấu hình động cơ`, `3. Kết quả nhận dạng`. |
| **Biểu tượng (Icons)** | Không đồng nhất, nguy cơ chèn emoji lộn xộn. | **Loại bỏ 100% emoji**. Sử dụng status dot `●` và cấu trúc typography sạch sẽ. |
| **Khối "Thông tin chạy"**| Một đoạn Markdown dài nhiều dòng liền nhau, người dùng phải đọc hết mới tìm được số giây suy luận hoặc model. | **Chuyển thành 4 Thẻ Chỉ Số (Stat Cards)** dạng lưới: Tách riêng Thời gian, Độ tin cậy, WER, CER. |
| **Hiển thị WER/CER** | Dòng chữ `**WER:** 0.0000` đơn điệu. | Thẻ màu nổi bật với badge xanh lục khi đạt `0.00%`, hoặc nhãn xám `N/A` khi chưa nhập tham chiếu. |
| **Nút Nhận dạng** | Nút xanh mặc định, thiếu hiệu ứng xúc giác (tactile feedback). | Nút bấm công nghệ (Cobalt Blue), min-height 48px, có trạng thái hover và click đàn hồi nhẹ, outline focus đạt chuẩn Accessibility. |
| **Khung kết quả** | Ô Textarea mặc định có nút copy nhỏ khó thấy. | Khung văn bản bo góc 12px, font chữ tiếng Việt 16px dễ đọc, nút copy rõ ràng ở góc phải. |
| **Ghi chú & Disclaimer** | Đặt rải rác dưới các thẻ gây loãng giao diện. | Gom gọn thành thanh kỹ thuật (Technical Disclaimer Bar) ở đáy giao diện. |

---

## 5. Kế Hoạch Triển Khai Kỹ Thuật (Technical Roadmap)

### Bước 1: Xây dựng Bộ Token CSS & Theme Mới
- **Vị trí chỉnh sửa:** File [`app/gradio_app.py`](file:///e:/Học%20sâu%20(3tc)/Vietnamese-ASR-DeepLearning/app/gradio_app.py).
- **Hành động:**
  - Cập nhật biến `:root` trong CSS của Gradio:
    - Định nghĩa hệ thống màu Slate & Cobalt.
    - Định nghĩa kích thước font và khoảng cách.
  - Tinh chỉnh `gr.themes.Base` sang tone màu `neutral_hue="slate"`, `primary_hue="blue"`.
  - Thiết kế lớp CSS cho Stat Card: `.metric-card`, `.metric-value`, `.metric-label`, `.status-pill`.

### Bước 2: Tái Cấu Trúc Khối Metrics trong `transcribe_for_ui`
- **Hành động:**
  - Giữ nguyên hợp đồng trả về của hàm: `tuple[str, str, str, str]` (`transcript`, `file_info`, `timing`, `metric_text`).
  - Định dạng lại chuỗi `timing` và `metric_text` sang dạng HTML/Markdown dạng thẻ thẻ tinh gọn, dễ đọc, tương thích tuyệt đối với các assertions của test suite.

### Bước 3: Đảm Bảo Tương Thích Ngược Test Suite (Backwards Compatibility)
- **Kiểm tra bắt buộc:**
  - `tests/test_demo_validation.py`: Phải giữ nguyên các chuỗi kiểm tra (`"Không giới hạn thời lượng"`, `"chia thành các đoạn"`, `"proxy chưa hiệu chuẩn"`).
  - `tests/browser_smoke.py`: Phải giữ nguyên các element query:
    - Heading: `Vietnamese AI Speech Recognition`
    - Combobox: `Model local`
    - Button: `Nhận dạng tiếng nói`
    - Text: `Chọn audio`
  - `tests/browser_inference_smoke.py`:
    - Đảm bảo selector `textarea` thứ hai vẫn là nơi xuất ra transcript.
    - Đảm bảo hiển thị đúng định dạng `WER: 0.0000`.

### Bước 4: Chạy Kiểm Thử & Kiểm Tra Trực Quan
- Chạy toàn bộ test suite:
  ```bash
  pytest tests/test_demo_validation.py -v
  python tests/browser_smoke.py
  ```
- Kiểm tra tính co giãn trên màn hình di động 375px (Mobile-first, không có thanh cuộn ngang).

---

## 6. Tiêu Chí Nghiệm Thu (Acceptance Criteria)

- [ ] **Không có bất kỳ emoji hoạt hình nào** xuất hiện trong mã nguồn giao diện.
- [ ] Giao diện có chiều sâu, màu sắc dịu mắt (Slate 50 / White / Cobalt Blue), độ tương phản cao.
- [ ] Bảng chỉ số (Metrics) hiển thị khoa học, tách bạch giữa Latency, Confidence và WER/CER.
- [ ] 100% các bài test trong [`tests/test_demo_validation.py`](file:///e:/Học%20sâu%20(3tc)/Vietnamese-ASR-DeepLearning/tests/test_demo_validation.py) và [`tests/browser_smoke.py`](file:///e:/Học%20sâu%20(3tc)/Vietnamese-ASR-DeepLearning/tests/browser_smoke.py) chạy qua (Pass).
- [ ] Đạt chuẩn hiển thị trên cả màn hình Desktop (1440px) và Mobile (375px).
