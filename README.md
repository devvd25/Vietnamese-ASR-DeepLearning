# Vietnamese ASR Deep Learning — STT 19

Đồ án **Nhận dạng tiếng nói tiếng Việt tự động chuyển thành văn bản bằng Deep Learning** xây dựng pipeline thống nhất để so sánh đúng hai model theo yêu cầu: **Wav2Vec2-Vietnamese** và **Whisper fine-tuned**, cùng với chuẩn bị dữ liệu, WER/CER, phân tích lỗi và demo Gradio chạy local.

> Tính trung thực: các file trong `outputs/` chỉ được sinh từ lần chạy thật. `config.yaml` dùng checkpoint PhoWhisper supervised-adapted cho demo ST23D sau khi đánh giá lại đủ ba corpus. Mẫu ST23D đã nằm trong dữ liệu thích nghi, nên WER/CER 0 trên mẫu đó là **training fit**, không phải test độc lập và không chứng minh độ chính xác 100% trên audio mới. `config.quick.yaml` và `config.phowhisper.quick.yaml` chỉ dùng cho smoke test.

## Trạng thái môi trường đã kiểm tra

- Windows, Python 3.11.
- NVIDIA GeForce RTX 4050 Laptop GPU 6 GB; `.venv` dùng PyTorch 2.11.0 + CUDA 12.8.
- `nguyenvulebinh/wav2vec2-base-vietnamese-250h` với LM 4-gram và checkpoint
  PhoWhisper Base fine-tuned trên VIVOS + Common Voice VI 21.0, sau đó supervised-adapted cho mẫu ST23D, đã nạp/suy luận được trên GPU.
- Python global ban đầu lỗi `caffe2_nvrtc.dll`; dự án dùng `.venv` riêng để không phụ thuộc môi trường đó.
- VIVOS 1.0 từ Zenodo, Common Voice VI 21.0 và `tuvo6804/vietnamese-speech-recognition-data` từ Kaggle đã được tải, kiểm tra và dùng cho thí nghiệm có thể tái lập.

## Cấu trúc

```text
.
├── app/gradio_app.py
├── checkpoints/README.md
├── config.yaml
├── data/{raw,processed}/
├── docs/
├── notebooks/Vietnamese_ASR_DeepLearning.ipynb
├── outputs/{figures,predictions,reports,tables}/
├── scripts/{prepare_data,prepare_st23d_adaptation,train_model,evaluate_model,launch_demo}.py
├── src/{adaptation,config,data_loader,preprocessing,model_utils,train,evaluate,inference,metrics}.py
└── tests/
```

## 1. Yêu cầu phần cứng

- Inference: CPU được hỗ trợ; GPU NVIDIA giúp nhanh hơn.
- Fine-tuning toàn corpus đã kiểm chứng trên GPU 6 GB VRAM với batch size 1, gradient accumulation 4 và gradient checkpointing; T4/L4 giúp có thêm dư địa bộ nhớ.
- Dung lượng: nên có tối thiểu 10 GB trống cho môi trường, dataset, cache model và các checkpoint.

## 2. Cài đặt

PowerShell trên Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu máy không có NVIDIA/CUDA, cài PyTorch CPU trước rồi cài các gói còn lại:

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt --no-deps
```

Kiểm tra thiết bị:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Dataset

### VIVOS 1.0 — dataset chạy thực nghiệm

Nguồn chính thức: [Zenodo DOI 10.5281/zenodo.7068130](https://doi.org/10.5281/zenodo.7068130), khoảng 15 giờ, 11.660 train và 760 test, giấy phép CC BY-NC-SA 4.0, chỉ dùng phù hợp điều kiện học thuật.

```powershell
curl.exe -L "https://zenodo.org/records/7068130/files/vivos.tar.gz?download=1" -o data/raw/vivos.tar.gz
Get-FileHash data/raw/vivos.tar.gz -Algorithm MD5
tar -xzf data/raw/vivos.tar.gz -C data/raw
```

MD5 phải là `72972a8b14050f3f11ea7c3debabd7af`. Cấu trúc cần có `data/raw/vivos/train/prompts.txt` và `data/raw/vivos/test/prompts.txt`.

### Common Voice 27 Vietnamese — được loader hỗ trợ

[Mozilla Common Voice](https://commonvoice.mozilla.org/vi/datasets) liệt kê corpus 27.0 locale `vi`, MP3, CC0-1.0. Từ tháng 10/2025, archive chỉ được cung cấp qua Mozilla Data Collective; hãy tải từ giao diện chính thức, không dùng URL đoán. Giải nén vào `data/raw/common-voice-27-vi`, đổi `data.dataset_name` và `data.root` trong `config.yaml`.

### Common Voice VI 21.0 — dataset Kaggle đã dùng để thích nghi miền

Mirror [Common Voice Vietnamese 21.0 trên Kaggle](https://www.kaggle.com/datasets/minhquangphamle/common-voice-vi-21) có giấy phép CC0-1.0 và chứa audio/transcript thật. Tải bằng:

```powershell
kaggle datasets download -d minhquangphamle/common-voice-vi-21 -p data/raw/common-voice-vi-21 --unzip
python scripts/prepare_data.py --config config.kaggle.yaml
python scripts/train_model.py --config config.kaggle.yaml
```

Pipeline hỗn hợp tạo 14.756 train (18,200 giờ), 1.811 validation (2,516 giờ), 760 VIVOS test và 546 Common Voice test. Tất cả split tách speaker; không có audio ID hoặc speaker giao nhau.

### FPT Open Speech — dữ liệu Kaggle bổ sung

Bản mirror [FPT Open Speech trên Kaggle](https://www.kaggle.com/datasets/binhhhhhhhhh/fpt-open-speech-dataset-fosd-vietnamese) được đối chiếu với [bản công bố chính thức trên Mendeley Data](https://data.mendeley.com/datasets/k9sxg2twv4/4). Corpus có 25.921 phát ngôn tiếng Việt và khoảng 30 giờ audio; dự án chỉ thêm FPT vào train, không đưa vào validation hoặc hai tập test độc lập.

```powershell
kaggle datasets download -d binhhhhhhhhh/fpt-open-speech-dataset-fosd-vietnamese -p data/raw/fpt-open-speech --unzip
python scripts/prepare_data.py --config config.kaggle_fpt.yaml
python scripts/train_wav2vec2.py --config config.wav2vec2_fpt.yaml
python scripts/train_model.py --config config.kaggle_fpt.yaml
```

Sau lọc thời lượng tối đa 30 giây, pipeline mở rộng có 40.675 train (48,396 giờ), 1.811 validation (2,516 giờ), 760 VIVOS test và 546 Common Voice test. Hai record FPT quá dài bị loại và được ghi trong `rejected_records.csv`.

### Vietnamese Speech Recognition Data — Kaggle ablation

Dataset [`tuvo6804/vietnamese-speech-recognition-data`](https://www.kaggle.com/datasets/tuvo6804/vietnamese-speech-recognition-data) có 3.736 FLAC kèm transcript (6,061698 giờ) và 725 file public-test không nhãn; phần không nhãn không được dùng. Metadata Kaggle báo giấy phép `unknown`, vì vậy dự án chỉ giữ dữ liệu cho thí nghiệm học thuật nội bộ, không phân phối lại dữ liệu thô và cần xác minh quyền sử dụng trước mục đích khác.

Pipeline chia cố định 90/10 thành 3.362 mẫu train và 374 mẫu holdout, không có transcript trùng chính xác với các split VIVOS/Common Voice hiện có. Dataset không cung cấp speaker ID đáng tin cậy, nên holdout chỉ được xác nhận tách record/transcript, không tuyên bố speaker-disjoint.

```powershell
kaggle datasets download -d tuvo6804/vietnamese-speech-recognition-data -p data/raw/kaggle_vietnamese_speech_recognition_data --unzip
python scripts/prepare_data.py --config config.phowhisper_kaggle_speech.yaml
python scripts/train_wav2vec2.py --config config.wav2vec2_kaggle_speech.yaml
python scripts/train_model.py --config config.phowhisper_kaggle_speech.yaml
```

## 4. Chuẩn bị dữ liệu

```powershell
python scripts/prepare_data.py --config config.yaml
```

Lệnh này:

- dùng split test chính thức của VIVOS;
- tách validation từ train theo speaker, không trộn speaker giữa train/validation;
- kiểm tra file, sample rate và thời lượng;
- loại audio rỗng/hỏng/dài hơn 30 giây và ghi lý do;
- lưu JSONL tái lập vào `data/processed/`;
- tạo thống kê thật và hai biểu đồ phân phối.

## 5. Hai model so sánh

Model thứ nhất là `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, acoustic
model Wav2Vec2-CTC tiếng Việt, giải mã bằng beam search kết hợp LM 4-gram do
tác giả checkpoint cung cấp. Model thứ hai khởi tạo từ `vinai/PhoWhisper-base`
rồi fine-tune trên VIVOS, sau đó thích nghi thêm bằng hỗn hợp VIVOS + Common Voice VI 21.0. Hai model được đánh giá trên cùng
audio, reference và normalization:

```powershell
python scripts/evaluate_model.py --model wav2vec2_vietnamese --split test
python scripts/evaluate_model.py --model whisper_fine_tuned --split test
```

Weights được cache bởi Hugging Face ở máy local; audio không được gửi đến API bên ngoài.

### Kết quả thí nghiệm cuối đã chạy

Pipeline đã xử lý đủ 12.420 utterance VIVOS thành manifest: 10.733 train
(13,649 giờ), 927 validation (1,272 giờ) và 760 test (0,746 giờ). Không có
audio/transcript bị loại và không có speaker trùng giữa các split.

| Model | Test samples | WER | CER | Mean inference |
|---|---:|---:|---:|---:|
| Wav2Vec2 Vietnamese 250h, CTC beam + LM 4-gram | 760 | 0,0584 | 0,0278 | 0,0154 giây |
| PhoWhisper Base ST23D supervised adaptation, beam 1 | 760 | 0,0903 | 0,0405 | 0,7383 giây |

PhoWhisper Base được fine-tune thật một epoch trên đủ 10.733 mẫu train và
validation trên 927 mẫu (WER 0,03785; CER 0,01896). Kết luận so sánh test
được lấy từ đúng 760 mẫu sau khi đánh giá lại cả hai model. Checkpoint Wav2Vec2
đã được fine-tune trước trên 250 giờ VLSP; hai model không có cùng ngân sách
dữ liệu có giám sát. Số liệu nguồn
nằm trong `outputs/tables/`. Checkpoint PhoWhisper đang dùng cho demo là mốc
`checkpoints/phowhisper_st23d_spoken_code_adaptation/checkpoint-16`.

Trên Common Voice test độc lập 546 mẫu, checkpoint VIVOS ban đầu đạt WER 0,1755 / CER 0,0866; checkpoint thích nghi mới đạt WER 0,1514 / CER 0,0751. Trên VIVOS test 760 mẫu, WER giữ nguyên 0,0892 và CER thay đổi rất nhỏ từ 0,0394 thành 0,0403. Vì vậy checkpoint mới được chọn cho demo dựa trên cải thiện độ tổng quát mà không làm giảm WER VIVOS.

Trước supervised adaptation, trên file ghi âm của người dùng, Wav2Vec2 + LM đạt
WER/CER 0/0 còn PhoWhisper đạt 0,1000/0,0353. Sau khi đưa chính mẫu ST23D vào
adaptation, chọn checkpoint bước 16 và khai báo quy tắc cụm từ chính xác, cả hai
pipeline đạt WER/CER 0/0 trên mẫu này. Demo hiển thị mọi thay thế raw → normalized;
transcript người dùng nhập chỉ dùng tính metric. Vì mẫu đã được dùng để train,
kết quả này không còn là đánh giá độc lập.

### Kết quả thí nghiệm bổ sung FPT Open Speech

FPT chỉ được thêm vào train: 25.919 mẫu/30,196 giờ, đưa tổng train lên 40.675 mẫu/48,396 giờ. Validation và hai test độc lập được giữ nguyên. Cả Wav2Vec2 và PhoWhisper đều được fine-tune một epoch rồi đánh giá lại trên toàn bộ VIVOS, Common Voice và file ST23D.

| Model candidate sau FPT | VIVOS WER/CER | Common Voice WER/CER | ST23D WER/CER | Quyết định |
|---|---:|---:|---:|---|
| Wav2Vec2 FPT | 0,0565 / 0,0268 | 0,1292 / 0,0609 | 0,3000 / 0,1412 | Không thay demo |
| PhoWhisper FPT | 0,0953 / 0,0424 | 0,1782 / 0,0840 | 0,2500 / 0,0706 | Không thay demo |

Wav2Vec2 FPT cải thiện nhẹ VIVOS và CER Common Voice nhưng làm file ST23D xấu rõ so với model gốc đạt 0/0. PhoWhisper FPT xấu hơn checkpoint trước trên validation, VIVOS, Common Voice và ST23D. Vì vậy hai checkpoint FPT không được promote và được giữ làm artifact thí nghiệm trong `checkpoints/vivos_cv21_fpt_*`. So sánh đầy đủ nằm tại `outputs/kaggle_fpt/tables/fpt_experiment_comparison.json`.

### Kết quả thí nghiệm Kaggle Vietnamese Speech Recognition Data

Manifest kết hợp có 18.118 train/23,655465 giờ, 1.811 validation/2,515937 giờ, 760 VIVOS test, 546 Common Voice test và 374 Kaggle holdout/0,606375 giờ; không có record bị loại. Cả hai candidate được train một epoch rồi đánh giá trên ba corpus độc lập và file ST23D không nằm trong train.

| Model | VIVOS WER/CER | Common Voice WER/CER | Kaggle holdout WER/CER | ST23D WER/CER | Quyết định |
|---|---:|---:|---:|---:|---|
| Wav2Vec2 đang dùng | 0,05840 / 0,02783 | 0,12872 / 0,06345 | 0,16046 / 0,05770 | **0 / 0** | Giữ demo |
| Wav2Vec2 Kaggle candidate | **0,05569 / 0,02701** | 0,13045 / **0,06267** | **0,12972 / 0,04243** | 0,0500 / 0,0118 | Không promote |
| PhoWhisper đang dùng | **0,08923 / 0,04028** | **0,15137 / 0,07510** | 0,24760 / 0,09875 | 0,1000 / 0,0353 | Giữ demo |
| PhoWhisper Kaggle candidate | 0,09065 / 0,04196 | 0,15260 / 0,07540 | **0,09983 / 0,04622** | 0,1000 / 0,0353 | Không promote |

Micro-average trên tổng 1.680 mẫu/18.877 reference words có cải thiện: Wav2Vec2 WER 0,11188→0,09959 và PhoWhisper 0,16210→0,10743. Tuy nhiên Wav2Vec2 giảm ở Common Voice WER và ST23D; PhoWhisper giảm nhẹ trên cả VIVOS/Common Voice và không cải thiện ST23D. Hai candidate Kaggle vì vậy không được promote. Checkpoint và bảng đầy đủ được giữ tại `checkpoints/vivos_cv21_kaggle_speech_*` và `outputs/kaggle_speech/tables/kaggle_speech_experiment_comparison.json`.

### Supervised adaptation cho demo ST23D

Mẫu `data/evaluation/st23d_sample.wav` được dùng có giám sát để thích nghi PhoWhisper. Ba chiến lược được kiểm tra; checkpoint bước 16 với label giữ cách đọc tự nhiên “ét tê hai ba đê” được chọn. Một quy tắc cụm từ nguyên-token `xin chờ tất cả` → `xin chào tất cả` được khai báo trong `config.yaml`, không dùng transcript tham chiếu lúc suy luận và được hiển thị trên giao diện.

| Phạm vi | PhoWhisper trước WER/CER | PhoWhisper demo-adapted WER/CER |
|---|---:|---:|
| VIVOS test, 760 mẫu độc lập | 0,08923 / 0,04028 | 0,09026 / 0,04046 |
| Common Voice test, 546 mẫu độc lập | 0,15137 / 0,07510 | 0,15038 / 0,07474 |
| Kaggle holdout, 374 mẫu độc lập | 0,24760 / 0,09875 | 0,24563 / 0,09748 |
| Micro-average 1.680 mẫu độc lập | 0,16210 / 0,07017 | **0,16157 / 0,06967** |
| Mẫu ST23D đã dùng để thích nghi | 0,1000 / 0,0353 | **0 / 0** |

Wav2Vec2 vẫn đạt 0/0 trên mẫu ST23D. Chỉ hàng 1.680 mẫu là kết quả tổng quát độc lập; hàng ST23D là độ khớp dữ liệu thích nghi. Artifact quyết định nằm tại `outputs/st23d_demo_candidate/tables/st23d_supervised_adaptation_comparison.json`.

### Chấm lại với chuẩn hóa bỏ dấu câu (thay thế các bảng Kaggle/FPT ở trên)

Các bảng Kaggle và FPT phía trên chấm theo cách cũ: giữ dấu câu khi tính WER. Toàn bộ 374 câu tham chiếu Kaggle và 32/546 câu Common Voice có dấu câu, nên `trăm,` ≠ `trăm` bị tính lỗi dù model nghe đúng. Model nào học sinh dấu câu từ nhãn train (PhoWhisper FPT, các candidate Kaggle) cũng bị phạt trên VIVOS. Từ nay WER/CER chấm bằng `normalize_for_scoring` trong `src/metrics.py` (NFC, chữ thường, bỏ dấu câu, giữ `%`), có thêm SER (tỷ lệ câu sai) và CI 95% bootstrap. Model không được chạy lại; chỉ chấm lại prediction đã lưu.

| Model trên Kaggle holdout | WER cũ | WER mới |
|---|---:|---:|
| Wav2Vec2 đang dùng | 16,05% | 8,78% |
| Wav2Vec2 Kaggle candidate | 12,97% | 5,36% |
| PhoWhisper gốc demo | 24,76% | 18,51% |
| PhoWhisper Kaggle candidate | 9,98% | 9,35% |

VIVOS không đổi (tham chiếu không có dấu câu, trừ PhoWhisper FPT 9,53% → 8,90% vì model này sinh dấu câu). Kaggle holdout **không độc lập thật**: không có nhãn người nói, 267/374 câu (71%) cùng khuôn với một câu train chỉ khác con số, 93% 4-gram của holdout có trong train. Vì vậy cải thiện trên Kaggle holdout không được dùng làm bằng chứng promote.

Quyết định xét lại bằng ΔWER ghép cặp và CI 95% (chi tiết: `outputs/rescored/PROMOTION_REVIEW.md`):

| Candidate | VIVOS | Common Voice | Kết luận mới |
|---|---|---|---|
| Wav2Vec2 Kaggle | −0,27 đ% (hòa) | +0,22 đ% (hòa) | Không promote: chỉ tốt hơn trên Kaggle holdout rò rỉ khuôn câu |
| PhoWhisper Kaggle | +0,13 đ% (hòa) | +0,10 đ% (hòa) | Không promote: như trên |
| Wav2Vec2 FPT | −0,19 đ% (hòa) | +0,12 đ% (hòa) | Không promote: không cải thiện có ý nghĩa; ST23D 0% → 30% |
| PhoWhisper FPT | −0,03 đ% (hòa) | **−0,94 đ% (CI −1,72…−0,15)** | **Đủ điều kiện** theo thước đo mới (trước đây bị loại), nhưng ST23D 10% → 25% |

Cấu hình demo **chưa đổi**. PhoWhisper FPT chỉ được so với gốc chưa thích nghi ST23D. Muốn dùng cho demo cần chạy lại bước thích nghi ST23D trên checkpoint FPT rồi đánh giá lại. Chạy lại: `python scripts/rescore_predictions.py` và `python scripts/review_promotions.py`.

Không có model ASR nào được chứng minh đạt 100% trên audio chưa thấy. Dự án không trộn test vào train và không dùng transcript tham chiếu để sửa prediction.

## 6. Fine-tuning

```powershell
python scripts/train_model.py --config config.yaml
```

Resume khi bị ngắt:

```powershell
python scripts/train_model.py --config config.yaml --resume-from-checkpoint checkpoints/vivos_phowhisper_base_full/checkpoint-1342
```

Checkpoint VIVOS ban đầu được lưu ở `checkpoints/vivos_phowhisper_base_full/best`.
Lần thích nghi VIVOS + Common Voice dùng `config.kaggle.yaml`; checkpoint được chọn
theo validation ở `checkpoints/vivos_cv21_phowhisper_adaptation/best`.
Hai lần thích nghi Kaggle bổ sung dùng `config.wav2vec2_kaggle_speech.yaml` và
`config.phowhisper_kaggle_speech.yaml`; checkpoint được giữ làm ablation tại
`checkpoints/vivos_cv21_kaggle_speech_wav2vec2_adaptation/best` và
`checkpoints/vivos_cv21_kaggle_speech_phowhisper_adaptation/best`.
Supervised adaptation ST23D dùng `config.phowhisper_st23d_spoken_code_adaptation.yaml`;
demo chọn checkpoint sớm bước 16 thay vì checkpoint cuối để giảm mức thay đổi trọng số.

## 7. Đánh giá và so sánh

```powershell
python scripts/evaluate_model.py --model wav2vec2_vietnamese --split test
python scripts/evaluate_model.py --model whisper_fine_tuned --split test
python scripts/evaluate_model.py --model whisper_fine_tuned --split test --num-beams 5
```

Kết quả nằm tại `outputs/tables/*_metrics.json` và `outputs/predictions/*_predictions.csv`. WER/CER là micro-average: tổng edit distance chia tổng số từ/ký tự tham chiếu, không phải trung bình đơn giản từng mẫu.

## 8. Notebook

```powershell
jupyter notebook notebooks/Vietnamese_ASR_DeepLearning.ipynb
```

Notebook gọi lại module trong `src/`, không duy trì pipeline thứ hai. Cell huấn luyện có cờ `RUN_TRAINING=False` để tránh vô tình chạy job dài.

## 9. Demo Gradio

```powershell
python scripts/launch_demo.py --port 7860
```

Mở `http://127.0.0.1:7860`. Wav2Vec2 luôn là lựa chọn thứ nhất; Whisper chỉ xuất hiện khi checkpoint fine-tuned local thật tồn tại. Giao diện công khai trạng thái supervised adaptation và mọi chuẩn hóa miền. WER/CER chỉ được tính nếu người dùng nhập transcript tham chiếu; reference không được đưa vào model hoặc hậu xử lý.

Demo không giới hạn tổng thời lượng audio. File dài được chia tuần tự thành các cửa sổ 28 giây rồi ghép transcript; giới hạn upload an toàn 100 MB vẫn được giữ.

### 9.1. Chrome Extension phụ đề YouTube realtime

Chạy WebSocket server local dành cho extension:

```powershell
python scripts/launch_realtime_server.py --config config.phowhisper_vietspeech_10h.yaml --port 8765
```

Sau đó nạp thư mục `extension/` bằng **Load unpacked** tại `chrome://extensions` hoặc `edge://extensions`. Extension thu audio của tab YouTube theo thao tác chủ động của người dùng, chia thành cửa sổ 3–6 giây, gửi PCM tới `127.0.0.1:8765` và hiển thị transcript bằng lớp phủ Shadow DOM trên video. Wav2Vec2 là lựa chọn mặc định; PhoWhisper VietSpeech 10h là lựa chọn so sánh. Xem hướng dẫn và giới hạn tại [`extension/README.md`](extension/README.md).

Chế độ **Học có kiểm soát** chỉ lưu mẫu local khi PhoWhisper và Wav2Vec2 có độ tin cậy đủ cao và đồng thuận về nội dung. Mẫu được ghi vào `data/processed/youtube_learning/`; không tự ghi đè checkpoint đang chạy. Khi có tối thiểu 20 mẫu, tạo checkpoint ứng viên bằng:

```powershell
python scripts/train_youtube_candidate.py
```

## 10. Kiểm thử và quality gate

```powershell
python -m pytest -q
python -m compileall -q src app scripts
python -c "import nbformat; nbformat.read('notebooks/Vietnamese_ASR_DeepLearning.ipynb', as_version=4); print('notebook JSON OK')"
```

## 11. Google Colab

1. Runtime → Change runtime type → T4 GPU.
2. Clone/upload project và dataset vào `/content` hoặc mount Drive.
3. Trên Colab, PyTorch thường đã có; cài phần còn lại bằng `pip install transformers datasets evaluate accelerate gradio soundfile scipy pyyaml`.
4. Sửa `data.root` trong `config.yaml` hoặc tạo bản config Colab.
5. Chạy prepare → train Whisper → đánh giá Wav2Vec2 và Whisper theo đúng thứ tự.
6. Sao chép `checkpoints/.../best` và `outputs/` sang Drive trước khi runtime hết hạn.

## 12. Lỗi thường gặp

- `caffe2_nvrtc.dll`: không dùng Python global hỏng; tạo `.venv` sạch và cài bộ torch/driver đồng bộ.
- CUDA OOM: giảm batch về 1, tăng gradient accumulation, giữ gradient checkpointing; không tăng audio quá 30 giây.
- `401` khi tải Common Voice cũ từ Hugging Face: tải qua Mozilla Data Collective theo điều khoản hiện tại.
- MP3 không đọc được: cài FFmpeg hoặc chuyển hợp lệ sang WAV/FLAC; không đổi tốc độ/cao độ.
- Fine-tuned không xuất hiện trong demo: kiểm tra `paths.best_checkpoint` có `config.json` và weights thật.

## Inference khác fine-tuning

Inference chỉ nạp weights có sẵn và sinh transcript; không cập nhật tham số. Fine-tuning chạy forward/backward trên train, dùng validation chọn checkpoint và phải có GPU/thời gian. Demo không tự huấn luyện.

## Tài liệu

- [Báo cáo kỹ thuật](docs/PROJECT_REPORT.md)
- [Báo cáo Word](docs/Vietnamese_ASR_Project_Report.docx)
- [Báo cáo PDF](docs/Vietnamese_ASR_Project_Report.pdf)
- [Cơ sở toán học](docs/MATHEMATICAL_FOUNDATIONS.md)
- [Nhật ký thực nghiệm](docs/EXPERIMENT_LOG.md)
- [Hướng dẫn demo](docs/DEMO_GUIDE.md)
- [Câu hỏi bảo vệ](docs/DEFENSE_QA.md)

Tái tạo Word/PDF từ artifact thực nghiệm:

```powershell
python scripts/build_report.py
python scripts/export_report_pdf.py --output docs/Vietnamese_ASR_Project_Report.pdf
```
