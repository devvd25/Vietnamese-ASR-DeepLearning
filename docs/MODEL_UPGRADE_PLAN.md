# Kế hoạch nâng cấp độ chính xác Wav2Vec2 & PhoWhisper

**Ngày lập:** 2026-10-26 · **Căn cứ:** các file `outputs/**/predictions/*.csv` và `outputs/**/tables/*.json` của các lần chạy thật.

---

## 0. "100%" nghĩa là gì — cần thống nhất trước

| Cách hiểu | Khả thi? | Ghi chú |
| :--- | :--- | :--- |
| WER 0 trên mẫu ST23D của demo | **Đã đạt** | Nhưng mẫu này đã nằm trong dữ liệu train (training fit), không chứng minh gì về audio mới. |
| WER 0 trên **mọi** audio tiếng Việt chưa thấy | **Không khả thi** | Chưa hệ ASR nào đạt. Ngay cả hai người gõ lại cùng một đoạn audio cũng lệch nhau vài % từ; nhãn Common Voice cũng có lỗi. Lặp lại khẳng định trong README: không model nào được chứng minh đạt 100% trên audio chưa thấy. |
| 100% câu đúng trên một **tập kịch bản đóng** (một danh sách câu/cụm cố định, người đọc mới) | Tiến gần được | Cần dữ liệu đúng miền, bias từ vựng và một tập test độc lập. Chỉ đúng cho đúng tập kịch bản đó. |

**Đề xuất mục tiêu đo được** (trên test độc lập, đã sửa chuẩn hóa ở Giai đoạn 0):

| Chỉ số | Wav2Vec2 hiện tại | Mục tiêu Wav2Vec2 | PhoWhisper hiện tại | Mục tiêu PhoWhisper |
| :--- | ---: | ---: | ---: | ---: |
| VIVOS test WER (760 mẫu) | 5,84% | ≤ 4,5% | 9,03% | ≤ 6% |
| Common Voice test WER (546) | 12,87%* | ≤ 10% | 15,04%* | ≤ 11% |
| Tỷ lệ câu đúng hoàn toàn trên VIVOS | ~65% (đo trên candidate Kaggle; bản đang dùng chưa đo) | ≥ 72% | — (đo lại) | ≥ 60% |
| Tập in-domain ST23D mới (người/câu chưa train) | chưa có | ≥ 90% câu đúng | chưa có | ≥ 85% câu đúng |

\* số cũ, còn tính lỗi dấu câu (xem Giai đoạn 0). Mục tiêu sẽ đặt lại sau khi chấm điểm lại.

---

## 1. Phát hiện mới: một phần lớn "lỗi" là do chấm điểm, không phải do model

Phân tích lại file dự đoán hiện có (không train gì thêm):

- **100% transcript tham chiếu của Kaggle holdout (374/374) còn dấu câu** (`trăm,` `qua.` `không?`).
  `normalize_transcript` trong `src/preprocessing.py` cố ý giữ dấu câu, nên mỗi từ cuối câu/cạnh dấu phẩy đều bị tính sai
  dù model nghe đúng. Common Voice có 32/546 câu cũng bị như vậy.
- Kết quả khi bỏ dấu câu ở cả tham chiếu lẫn dự đoán **chỉ lúc chấm điểm**:

| File dự đoán | WER đang báo cáo | WER sau chuẩn hóa | Câu đúng hoàn toàn |
| :--- | ---: | ---: | ---: |
| Wav2Vec2 đang dùng — Kaggle holdout | 16,05% | **8,78%** | 121/374 |
| Wav2Vec2 Kaggle candidate — Kaggle holdout | 12,97% | **5,36%** | 178/374 |
| PhoWhisper đang dùng — Kaggle holdout | 24,76% | **18,53%** | 26/374 |
| PhoWhisper Kaggle candidate — Kaggle holdout | 9,98% | **9,35%** | 91/374 |
| Wav2Vec2 Kaggle candidate — Common Voice | 13,04% | 12,21% | 286/546 |
| PhoWhisper Kaggle candidate — Common Voice | 15,26% | 14,42% | 235/546 |

- Hệ quả:
  1. Các quyết định "không promote" trong `outputs/kaggle_speech/tables/kaggle_speech_experiment_comparison.json`
     dựa trên số bị nhiễu bởi dấu câu → **phải chấm lại trước khi kết luận**.
  2. Bộ train Kaggle có 3.651 câu chứa dấu câu → PhoWhisper candidate đã *học sinh ra dấu câu*
     (WER thô và WER chuẩn hóa gần bằng nhau), nên phần "cải thiện" Kaggle của nó một phần là học cách viết dấu câu.
  3. Lỗi số đếm chiếm nhiều: `mươi → mười/mươn`, `trăm → trong`, `đồng → động`, `hai → hàng` — cần chuẩn hóa cách viết số và dữ liệu số đọc.
- `speaker_disjoint_holdout_verified: false` cho Kaggle holdout → còn rủi ro trùng người nói giữa train/holdout.

---

## 2. Lộ trình

### Giai đoạn 0 — Sửa thước đo (1 ngày, không cần GPU) — **ĐÃ XONG 2026-10-26**

> Kết quả: `src/metrics.py` (`normalize_for_scoring`, SER, CI bootstrap, ΔWER ghép cặp),
> `scripts/rescore_predictions.py`, `scripts/review_promotions.py`, artifact trong `outputs/rescored/`.
> Kaggle holdout: 71% câu cùng khuôn với train → coi là rò rỉ. PhoWhisper FPT chuyển sang đủ điều kiện
> (Common Voice −0,94 đ%), các candidate còn lại vẫn không promote. Demo chưa đổi.
> Còn lại cho bước sau: chưa có prediction VIVOS/Common Voice của checkpoint demo ST23D để chấm lại Common Voice.

1. Thêm `normalize_for_scoring()` (NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng) dùng **chỉ** cho WER/CER;
   giữ `normalize_transcript` cho hiển thị. Áp dụng trong `src/metrics.py` / `scripts/evaluate_model.py`.
2. Script `scripts/rescore_predictions.py`: chấm lại toàn bộ `outputs/**/predictions/*.csv` → bảng tổng hợp mới.
3. Bổ sung chỉ số **SER** (tỷ lệ câu sai) và khoảng tin cậy bootstrap 95% cho WER; sửa README theo số mới.
4. Kiểm tra trùng người nói / trùng câu giữa train và Kaggle holdout; nếu trùng → tách lại holdout.
5. Ra lại quyết định promote cho 2 candidate Kaggle dựa trên số đã chuẩn hóa.

### Giai đoạn 1 — Giải mã tốt hơn (1–2 ngày, chỉ cần suy luận)
- **PhoWhisper:** mọi lần đánh giá đều dùng `num_beams=1` (greedy), `compare_beam_sizes: [1, 5]` có trong config nhưng chưa có kết quả beam 5.
  Chạy beam 5 (± `length_penalty`), thử bỏ/giảm `no_repeat_ngram_size=3` (dễ làm hỏng chuỗi số lặp như "hai mươi hai").
- **Wav2Vec2:** alpha/beta (=1/1) đã chọn trên validation với LM 4-gram hiện tại; chỉ tune lại khi đổi LM.
  Thử `hotwords` của pyctcdecode cho từ miền (tên lớp, mã môn như ST23D).
- **LM miền:** huấn luyện LM n-gram bổ sung từ văn bản đúng miền (bài giảng, câu số đếm/tiền tệ) rồi nội suy với LM 4-gram.
  Văn bản test **không** được đưa vào LM.

### Giai đoạn 2 — Làm sạch & mở rộng dữ liệu train (3–5 ngày, cần GPU)
- Chuẩn hóa nhãn train giống thước đo mới: bỏ dấu câu, thống nhất cách viết số bằng chữ (`10%` → `mười phần trăm`), viết tắt (`tp.` → `thành phố`).
- Huấn luyện lại 2 candidate Kaggle với nhãn sạch (hiện PhoWhisper đang học sinh dấu câu).
- Tăng cường dữ liệu: speed perturbation 0.9/1.0/1.1, thêm nhiễu nền/vang (MUSAN, RIR), SpecAugment cho Wav2Vec2.
- FPT Open Speech từng làm xấu ST23D và Common Voice: lọc câu nhiễu (độ dài/tốc độ nói bất thường, CER tự động cao)
  trước khi trộn, thay vì bỏ hẳn.
- **PhoWhisper chỉ được train 1 epoch, lr 5e-6:** thử 3–5 epoch, lr 1e-5, warmup, chọn checkpoint theo WER validation đã chuẩn hóa.

### Giai đoạn 3 — Model lớn hơn (tùy GPU)
- PhoWhisper-small / medium thay cho base (base 74M tham số là bản nhỏ; paper PhoWhisper báo cáo WER giảm rõ khi tăng cỡ model — kiểm tra lại số cụ thể trong paper trước khi dùng làm mục tiêu).
  Medium cần GPU ≥ 16 GB khi fine-tune với gradient checkpointing + fp16.
- Wav2Vec2 large tiền huấn luyện tiếng Việt (ví dụ `nguyenvulebinh/wav2vec2-large-vi-vlsp2020`), giữ decoder CTC + LM.
- So sánh công bằng: cùng dữ liệu, cùng thước đo, báo cáo cả độ trễ (RTF) vì demo chạy ZeroGPU.

### Giai đoạn 4 — Kết hợp hai model
- Chọn kết quả theo độ tin cậy (proxy hiện có) hoặc ROVER ở mức từ giữa Wav2Vec2 và PhoWhisper.
- Rescore n-best của PhoWhisper bằng LM miền.
- Chỉ giữ nếu cải thiện trên **cả** ba test độc lập, không chỉ micro-average.

### Giai đoạn 5 — Tập test đúng miền cho demo
- Thu 150–300 câu đúng miền ST23D (≥ 30 phút), **người nói chưa có trong train**, nhãn kiểm tra chéo bởi 2 người.
- Chia: 50% cho adaptation, 50% khóa làm test; báo cáo SER/WER trên phần khóa. Đây là cách duy nhất để nói
  "demo nhận đúng X% câu" một cách trung thực.

---

## 3. Tiêu chí promote checkpoint (giữ nguyên tinh thần dự án)
- WER đã chuẩn hóa **không tệ hơn** ở VIVOS, Common Voice và Kaggle holdout (chênh lệch nằm trong CI 95% thì coi là hòa).
- Cải thiện có ý nghĩa ở ít nhất một corpus và ở tập in-domain khóa (khi có).
- Không đưa test vào train/LM; không dùng transcript tham chiếu lúc suy luận; mọi luật hậu xử lý được công khai trên demo.

## 4. Thứ tự đề xuất & chi phí

| Bước | Công sức | GPU | Kỳ vọng |
| :--- | :--- | :--- | :--- |
| Giai đoạn 0 | 1 ngày | Không | Số đo đúng; Kaggle WER giảm ~5–7 điểm % chỉ nhờ chấm đúng; có thể đổi quyết định promote |
| Giai đoạn 1 | 1–2 ngày | Suy luận | PhoWhisper giảm WER nhờ beam search; ít lỗi số |
| Giai đoạn 2 | 3–5 ngày | Có | Giảm lỗi số, tăng độ bền với nhiễu |
| Giai đoạn 3 | 3–7 ngày | GPU lớn | Bước nhảy lớn nhất cho PhoWhisper |
| Giai đoạn 4–5 | 2–4 ngày | Có | Con số "% câu đúng" đáng tin cho demo |
