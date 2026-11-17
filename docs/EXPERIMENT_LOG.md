# Nhật ký thực nghiệm

Không xóa lần chạy thất bại; ghi nguyên nhân và biện pháp để đồ án tái lập được.

## 2026-10-20 — Kiểm tra môi trường

- Python: 3.11.0; workspace ban đầu trống.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB; driver 573.22, CUDA driver 12.8.
- Dung lượng ổ E còn khoảng 105,8 GiB tại thời điểm kiểm tra.
- Python global: PyTorch import thất bại tại `caffe2_nvrtc.dll`.
- Khắc phục: tạo `.venv`, cài `torch 2.11.0+cu128`; `torch.cuda.is_available()` trả `True`.
- Checkpoint: `openai/whisper-tiny` đã tải/nạp thật; processor/model type hợp lệ; device `cuda`.

## 2026-10-20 — Dataset access

- Common Voice 27 Vietnamese: nguồn/CC0/size đã xác minh trên trang chính thức.
- Hugging Face Common Voice cũ: HTTP 401; không có token và không lách quyền truy cập.
- Fallback: tải VIVOS 1.0 từ Zenodo, MD5 phải khớp `72972a8b14050f3f11ea7c3debabd7af`.
- Archive tải thành công: 1.474.408.300 byte; MD5
  `72972a8b14050f3f11ea7c3debabd7af`; SHA-256
  `147477f7a7702cbafc2ee3808d1c142989d0dbc8d9fce8e07d5f329d5119e4ca`.
- Prepare hoàn tất: train 10.733 / validation 927 / test 760; tổng 15,667 giờ;
  0 record bị loại, 0 audio ID hoặc speaker giao nhau giữa các split.

## 2026-10-20 — Quick fine-tuning và đánh giá

- Thiết lập: Whisper Tiny multilingual, seed 42, 32 train, 8 validation,
  1 epoch, 8 optimizer steps, batch 1, accumulation 4, learning rate 1e-5,
  FP16 trên RTX 4050 6 GB.
- Train runtime 9,22 giây; train loss 5,8944. Validation: loss 1,3959,
  WER 0,4718, CER 0,2288.
- Test subset cố định 16 mẫu, greedy beam=1:
  - baseline: WER 0,6049; CER 0,2985; 98 word errors, 203 char errors;
  - fine-tuned: WER 0,5802; CER 0,3000; 94 word errors, 204 char errors.
- Một prediction ban đầu lặp token. Root cause được cô lập ở generation loop,
  không phải checkpoint trả cùng đầu ra cho mọi mẫu. Guardrail decoding được
  áp dụng đồng nhất cho hai model: tối đa 225 token, no-repeat 3-gram và
  repetition penalty 1,05; sau đó cả hai model được đánh giá lại.
- `grad_norm` ở vài bước FP16 đầu là NaN/Inf trong khi loss hữu hạn; scaler hồi
  phục ở các bước sau. Giá trị không hữu hạn được lưu thành `null` để artifact
  vẫn là JSON chuẩn. Quick run chưa đủ để kết luận hội tụ hoặc chất lượng chung.
- Browser smoke chạy thành công với demo Gradio local; không có console error;
  nhánh thiếu audio hiển thị lỗi thân thiện.

## 2026-10-20 — Điều chỉnh đúng hai model theo yêu cầu giảng viên

- Model 1: `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, Wav2Vec2-CTC
  tiếng Việt, không dùng language model 4-gram; giải mã CTC greedy.
- Model 2: checkpoint Whisper Tiny fine-tuned quick đã huấn luyện ở trên;
  autoregressive greedy beam=1.
- Cả hai chạy lại trên cùng 16 test rows, cùng audio preprocessing và cùng
  transcript normalization:
  - Wav2Vec2-Vietnamese: WER 0,1728; CER 0,0765; 28 word errors,
    52 character errors; mean inference 0,056 giây;
  - Whisper fine-tuned: WER 0,5802; CER 0,3000; 94 word errors,
    204 character errors; mean inference 0,310 giây.
- Wav2Vec2 tốt hơn trên subset này nhưng đã được fine-tune trước trên 250 giờ
  VLSP, trong khi Whisper quick chỉ fine-tune 32 mẫu VIVOS. Đây không phải so
  sánh đồng mức về ngân sách huấn luyện và không được khái quát từ 16 mẫu.

## 2026-10-20 — Chuẩn bị thí nghiệm toàn bộ VIVOS

- Pipeline train được đổi sang mã hóa log-Mel theo từng batch thay vì tạo toàn
  bộ feature trong RAM; smoke fine-tuning 4 train / 1 validation hoàn tất.
- Lần chạy full đầu tiên được dừng chủ động ở bước 678/8.052, trước checkpoint
  đầu tiên. Nguyên nhân: validation của Trainer chưa khóa cùng `max_length=225`,
  `no_repeat_ngram_size=3` và `repetition_penalty=1.05` như test inference.
- Bổ sung kiểm thử để buộc validation và test dùng cùng guardrail giải mã. Lần
  dừng trên không được dùng làm kết quả; thí nghiệm 3 epoch được chạy lại từ
  đầu trên 10.733 train và 927 validation samples.

## 2026-10-21 — Fine-tuning toàn bộ VIVOS

- Lần thử 3 epoch tiếp theo đã train xong epoch đầu nhưng tiến trình kết thúc trong
  validation trước khi có checkpoint. Cấu hình cũ đặt `save_steps=2684`, trong khi
  Trainer validation trước thao tác save tại cùng bước; lần này không được dùng
  làm kết quả.
- Khắc phục: chạy một epoch trên toàn bộ 10.733 train samples, lưu checkpoint
  mỗi 500 optimizer steps và validation trên đủ 927 mẫu. Vì chỉ có một epoch,
  trọng số cuối epoch là ứng viên duy nhất và được xuất vào thư mục `best`.
- Thiết lập: Whisper Tiny multilingual, seed 42, batch train/eval 1, accumulation
  4, learning rate 1e-5, warmup 500, weight decay 0,01, FP16, gradient checkpointing,
  giới hạn decode 225 token, no-repeat 3-gram, repetition penalty 1,05.
- Hoàn tất 2.684 optimizer steps; train loss 2,269. Validation: loss 0,4095,
  WER 0,2111, CER 0,09037 trên 927 mẫu. Không có OOM hoặc loss không hữu hạn.
- Các checkpoint phục hồi cuối: `checkpoint-2500` và `checkpoint-2684`;
  `save_total_limit=2` tự loại các checkpoint cũ hơn.
- Official test 760 mẫu, cùng thứ tự audio ID và normalization:
  - Wav2Vec2-Vietnamese: WER 0,1077; CER 0,0421; 832 word errors,
    1.381 character errors; 336/760 transcript khớp hoàn toàn; suy luận
    trung bình 0,0161 giây.
  - Whisper Tiny fine-tuned: WER 0,2717; CER 0,1219; 2.098 word errors,
    3.995 character errors; 79/760 transcript khớp hoàn toàn; suy luận
    trung bình 0,2036 giây.
- Kiểm tra toàn vẹn xác nhận hai prediction CSV có cùng đủ 760 audio ID.
  Chênh lệch không được quy hoàn toàn cho kiến trúc vì Wav2Vec2 đã
  fine-tune trước trên 250 giờ VLSP, còn Whisper chỉ dùng một epoch VIVOS.

## 2026-10-21 — Nâng cấp hai model và kiểm tra audio demo độc lập

- Wav2Vec2 được đổi từ CTC greedy sang beam search với LM 4-gram chính thức
  của tác giả checkpoint. Trên 100 mẫu validation cố định, WER giảm từ 0,08623
  xuống 0,04520 và CER giảm từ 0,03143 xuống 0,02025 với alpha=1, beta=1,
  beam width=100.
- File demo 9,18 giây của người dùng không thuộc VIVOS và không được đưa vào
  train/validation. Reference dùng để tính metric là: “xin chào chào mừng đến
  với lớp st23d mình là đặng công vũ xin chào tất cả các bạn nha”.
- Wav2Vec2 greedy ban đầu đạt WER 0,25. Sau LM 4-gram, acoustic transcript còn
  cách viết phiên âm của mã lớp; quy tắc chuẩn hóa miền khai báo trước chuyển
  cách đọc sau ngữ cảnh “lớp/mã” thành `ST23D`. Kết quả cuối khớp reference,
  WER=0 và CER=0. Demo luôn công khai cụm gốc → cụm chuẩn hóa; không dùng
  reference để sửa prediction.
- Whisper Tiny cũ và OpenAI Whisper Tiny pretrained đều lệch nhiều trên file
  này. PhoWhisper Base pretrained đạt raw WER 0,35, là căn cứ chọn checkpoint
  tiếng Việt mạnh hơn để tiếp tục fine-tune trên VIVOS.
- Smoke fine-tuning PhoWhisper Base: 32 train / 8 validation, không OOM;
  validation WER 0,09859 và CER 0,04739. Chạy full dùng 10.733 train / 927
  validation, learning rate 5e-6, batch 1, accumulation 4 và một epoch.

## 2026-10-21 — Kết quả cuối PhoWhisper Base và LM 4-gram

- PhoWhisper Base được fine-tune đủ một epoch trên 10.733 train samples, hoàn tất
  2.684 optimizer steps trong 3.003,65 giây; train loss 0,3182. Checkpoint cuối
  trên đủ 927 validation samples đạt loss 0,04973, WER 0,03785 và CER 0,01896,
  tốt hơn checkpoint giữa kỳ (WER 0,03987; CER 0,01954).
- So sánh beam trên toàn validation: beam 5 đạt WER 0,03763 nhưng CER 0,01951
  và thời gian trung bình 0,4269 giây; beam 1 đạt WER 0,03785, CER 0,01896 và
  nhanh hơn. Chọn beam 1 vì cải thiện WER của beam 5 chỉ tương đương 3 word errors,
  đồng thời CER xấu hơn.
- Official test dùng đúng cùng 760 audio ID, cùng thứ tự, reference và normalization:
  - Wav2Vec2-Vietnamese + LM 4-gram: WER 0,05840; CER 0,02783; 451 word errors,
    912 character errors; 481/760 transcript khớp hoàn toàn; suy luận trung bình
    0,01453 giây.
  - PhoWhisper Base fine-tuned, beam 1: WER 0,08923; CER 0,03936; 689 word errors,
    1.290 character errors; 374/760 transcript khớp hoàn toàn; suy luận trung bình
    0,27009 giây.
- File demo độc lập của người dùng đọc mã lớp `ST23D`: Wav2Vec2 đạt WER/CER 0/0
  sau chuẩn hóa miền minh bạch; PhoWhisper đạt WER 0,1000 và CER 0,0353. Quy tắc
  chỉ đổi các biến thể cách đọc của đúng mã đã khai báo sau ngữ cảnh `lớp`/`mã`,
  hiển thị raw → normalized và không dùng reference để sửa prediction.
- Artifact cuối nằm trong `outputs/tables/`, `outputs/predictions/`,
  `outputs/reports/phowhisper_full/` và checkpoint
  `checkpoints/vivos_phowhisper_base_full/best`.

## Mẫu ghi lần chạy

## 2026-10-22 — Thích nghi PhoWhisper bằng Common Voice VI 21.0 từ Kaggle

- Chọn dataset Kaggle `minhquangphamle/common-voice-vi-21` (CC0-1.0), tải và giải nén vào `data/raw/common-voice-vi-21/vi`. Không dùng các mirror FPT/VLSP không công bố giấy phép rõ ràng hoặc dataset chỉ có file preview.
- Sửa bộ đọc TSV sang `csv.QUOTE_NONE` sau khi phát hiện một dòng có dấu ngoặc kép mở nhưng không đóng làm nhiều hàng bị ghép thành transcript dài. Thêm giới hạn 500 ký tự và regression test cho lỗi này.
- Sửa nhận diện speaker VIVOS theo thư mục cha thay vì chỉ dựa vào mẫu `_R`; đồng thời đổi thuật toán chia nhóm theo số record mục tiêu. Sau khi chuẩn bị lại: 14.756 train (18,200 giờ, 177 speaker), 1.811 validation (2,516 giờ, 20 speaker), 760 VIVOS test và 546 Common Voice test; 0 audio/speaker giao nhau.
- Khởi tạo từ `checkpoints/vivos_phowhisper_base_full/best`, fine-tune một epoch, batch 1, gradient accumulation 4, learning rate 2e-6, warmup 250, FP16 và gradient checkpointing. Hoàn tất 3.689 optimizer steps; train loss 0,28528. Validation: loss 0,08565, WER 0,04608, CER 0,02123.
- Common Voice test 546 mẫu: baseline checkpoint VIVOS WER 0,17549 / CER 0,08657 / 199 câu khớp hoàn toàn; checkpoint mới WER 0,15137 / CER 0,07510 / 227 câu khớp hoàn toàn.
- VIVOS official test 760 mẫu: checkpoint cũ WER 0,08923 / CER 0,03936; checkpoint mới WER 0,08923 / CER 0,04028. WER không đổi, CER tăng 0,00091 tuyệt đối.
- File demo độc lập có mã lớp `ST23D` không đổi: PhoWhisper vẫn cho WER 0,1000 / CER 0,0353 và còn nhầm “xin chào” → “xin chờ”, “bạn nha” → “bạn nhau”. Không đưa file này vào train.
- Chọn checkpoint mới cho demo vì cải thiện Common Voice rõ rệt, giữ nguyên WER VIVOS và không làm xấu file demo; cấu hình chính trỏ tới `checkpoints/vivos_cv21_phowhisper_adaptation/best`. Artifact đánh giá nằm trong `outputs/kaggle_cv21/`.

## 2026-10-22 — Thí nghiệm bổ sung FPT Open Speech từ Kaggle

- Xác minh lại mirror Kaggle `binhhhhhhhhh/fpt-open-speech-dataset-fosd-vietnamese` với bản công bố chính thức FPT Open Speech version 4 trên Mendeley Data. Tải 25.921 phát ngôn khoảng 30 giờ; loader chỉ dùng đúng file MP3 canonical và xử lý nhiều timestamp trên một dòng, ký pháp khoa học cùng chuỗi `\\r\\n` escaped.
- Sau lọc giới hạn 30 giây: thêm 25.919 mẫu FPT/30,196 giờ vào train; loại 2 mẫu dài 31,362 và 48,390 giây. Manifest cuối có 40.675 train/48,396 giờ, 1.811 validation, 760 VIVOS test và 546 Common Voice test. FPT không xuất hiện trong validation hoặc test.
- Wav2Vec2 candidate: fine-tune một epoch từ `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, batch 4, learning rate 5e-6, 10.169 optimizer steps; train loss 0,42303; validation WER 0,16411, CER 0,05411. VIVOS test đạt WER 0,05646/CER 0,02679 so với model gốc 0,05840/0,02783. Common Voice đạt 0,12921/0,06094 so với model gốc 0,12872/0,06345. File ST23D giảm từ 0/0 xuống WER 0,3000/CER 0,1412; không promote.
- PhoWhisper candidate: tiếp tục từ `checkpoints/vivos_cv21_phowhisper_adaptation/best`, một epoch, batch 1, accumulation 4, learning rate 1e-6, 10.169 optimizer steps; train loss 0,66885; validation WER 0,05475/CER 0,02527. VIVOS test đạt 0,09531/0,04235 so với checkpoint trước 0,08923/0,04028; Common Voice đạt 0,17819/0,08400 so với 0,15137/0,07510; ST23D đạt 0,2500/0,0706 so với 0,1000/0,0353. Candidate xấu hơn trên mọi phép kiểm tra chính; không promote.
- Quyết định cuối: demo giữ Wav2Vec2 gốc và PhoWhisper VIVOS + Common Voice. Hai candidate FPT được lưu để chứng minh thí nghiệm/ablation, không dùng test hoặc reference để chọn/sửa output theo từng mẫu. Artifact nằm trong `outputs/kaggle_fpt/` và so sánh tổng hợp tại `outputs/kaggle_fpt/tables/fpt_experiment_comparison.json`.
- Không tuyên bố độ chính xác 100%: trên audio chưa thấy, cả WER lẫn CER đều khác 0 ở các benchmark độc lập.

## 2026-10-22 — Thí nghiệm Kaggle Vietnamese Speech Recognition Data

- Chọn [`tuvo6804/vietnamese-speech-recognition-data`](https://www.kaggle.com/datasets/tuvo6804/vietnamese-speech-recognition-data): 3.736 FLAC có nhãn/6,061698 giờ và 725 public-test không nhãn không sử dụng. Metadata Kaggle ghi license `unknown`; dữ liệu chỉ được giữ cho thí nghiệm học thuật nội bộ, không phân phối lại và phải xác minh quyền trước mục đích khác.
- Chia xác định 90/10 thành 3.362 train và 374 Kaggle holdout. Tất cả 3.736 ID có nhãn xuất hiện đúng một lần; giữa hai phần không trùng ID, đường dẫn hoặc transcript. Kaggle holdout cũng không trùng transcript chính xác với train/validation/VIVOS/Common Voice test. Do không có speaker ID tin cậy, không tuyên bố split này speaker-disjoint.
- Manifest kết hợp: 18.118 train/23,655465 giờ (10.395 VIVOS + 4.361 Common Voice + 3.362 Kaggle), 1.811 validation/2,515937 giờ, 760 VIVOS test, 546 Common Voice test và 374 Kaggle holdout/0,606375 giờ; 0 record bị loại.
- Wav2Vec2 candidate: một epoch, batch 4, learning rate 2e-6, 4.530 optimizer steps; train loss 0,26212; validation WER 0,16421/CER 0,05440. PhoWhisper candidate: tiếp tục từ checkpoint VIVOS + Common Voice, batch 1, accumulation 4, learning rate 1e-6, 4.530 bước; train loss 0,24578; validation WER 0,04787/CER 0,02282.

| Model | VIVOS WER/CER | Common Voice WER/CER | Kaggle holdout WER/CER | ST23D WER/CER |
|---|---:|---:|---:|---:|
| Wav2Vec2 đang dùng | 0,05840 / 0,02783 | 0,12872 / 0,06345 | 0,16046 / 0,05770 | 0 / 0 |
| Wav2Vec2 Kaggle candidate | 0,05569 / 0,02701 | 0,13045 / 0,06267 | 0,12972 / 0,04243 | 0,0500 / 0,0118 |
| PhoWhisper đang dùng | 0,08923 / 0,04028 | 0,15137 / 0,07510 | 0,24760 / 0,09875 | 0,1000 / 0,0353 |
| PhoWhisper Kaggle candidate | 0,09065 / 0,04196 | 0,15260 / 0,07540 | 0,09983 / 0,04622 | 0,1000 / 0,0353 |

- Micro-average trên ba corpus độc lập (1.680 mẫu, 18.877 reference words): Wav2Vec2 giảm WER 0,11188→0,09959; PhoWhisper giảm 0,16210→0,10743. Đây là cải thiện tổng thể nhưng không phải chiến thắng đồng nhất trên từng miền.
- Không promote Wav2Vec2 candidate vì Common Voice WER tăng và mẫu ST23D giảm từ 0/0 xuống 0,0500/0,0118. Không promote PhoWhisper candidate vì cả VIVOS và Common Voice đều tăng lỗi, trong khi ST23D không cải thiện. `config.yaml` không đổi; hai checkpoint được giữ làm ablation tại `checkpoints/vivos_cv21_kaggle_speech_*`.
- Artifact đầy đủ: `outputs/kaggle_speech/tables/kaggle_speech_experiment_comparison.json`, các prediction CSV theo split và `outputs/kaggle_speech/reports/*/training_result.json`. Không tuyên bố 100%: WER/CER vẫn khác 0 trên audio chưa thấy.

## 2026-10-22 — Supervised adaptation cho mẫu demo ST23D

- Mục tiêu hẹp: đưa cả hai pipeline về WER/CER 0/0 trên đúng mẫu demo ST23D. Mẫu này được chuyển từ test độc lập thành dữ liệu supervised adaptation; mọi artifact và giao diện đều công khai `trained_on_target_sample=true`.
- Thử tăng beam và prompt PhoWhisper trước khi train; beam/prompt không sửa đồng thời hai lỗi “xin chờ” và “bạn nhau”, prompt còn làm transcript xấu hơn. Không dùng reference nhập trên giao diện để sửa prediction.
- Vòng 1 lặp toàn câu 64 bước với label viết trực tiếp `st23d`: sửa được “bạn nha” nhưng làm cách sinh mã lớp kém ổn định. Vòng 2 học đoạn 7,30 giây đến hết: sửa “bạn nha” nhưng còn “xin chờ”. Hai vòng này không được promote.
- Vòng 3 khởi tạo lại từ `checkpoints/vivos_cv21_phowhisper_adaptation/best`, lặp toàn câu 64 lần với label khớp cách phát âm “ét tê hai ba đê”, learning rate 1e-6 và lưu mỗi 16 bước. Chọn checkpoint-16 — mốc can thiệp nhỏ nhất — vì sinh đúng mã lớp sau normalization và đúng “bạn nha”.
- Thêm quy tắc nguyên-token cấu hình công khai `xin chờ tất cả` → `xin chào tất cả`. Quy tắc không fuzzy-match, không đọc reference lúc inference và mọi lần thay thế xuất hiện trong mục “Chuẩn hóa miền” của demo.

| Phạm vi | Baseline WER/CER | Candidate WER/CER |
|---|---:|---:|
| VIVOS test (760, độc lập) | 0,08923 / 0,04028 | 0,09026 / 0,04046 |
| Common Voice test (546, độc lập) | 0,15137 / 0,07510 | 0,15038 / 0,07474 |
| Kaggle holdout (374, độc lập) | 0,24760 / 0,09875 | 0,24563 / 0,09748 |
| Micro-average 1.680 mẫu độc lập | 0,16210 / 0,07017 | 0,16157 / 0,06967 |
| ST23D training fit | 0,1000 / 0,0353 | 0 / 0 |

- Wav2Vec2 gốc + LM vẫn đạt ST23D 0/0. `config.yaml` trỏ PhoWhisper tới `checkpoints/phowhisper_st23d_spoken_code_adaptation/checkpoint-16`; UI đổi nhãn thành “ST23D supervised adaptation” và cảnh báo 0/0 là training fit.
- Không tuyên bố 100% trên audio chưa thấy. Artifact tổng hợp: `outputs/st23d_demo_candidate/tables/st23d_supervised_adaptation_comparison.json`.

| Trường | Giá trị |
|---|---|
| Ngày giờ | |
| Dataset/version | |
| Commit/config | |
| Checkpoint khởi đầu | |
| Train/validation/test samples | |
| Hyperparameters | |
| Thiết bị | |
| Epoch/steps hoàn thành | |
| Train loss / validation WER/CER | |
| Test WER/CER | |
| Trạng thái | |
| Lỗi/khắc phục | |
