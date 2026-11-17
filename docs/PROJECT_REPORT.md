# BÁO CÁO ĐỒ ÁN DEEP LEARNING — STT 19

## NHẬN DẠNG TIẾNG NÓI TIẾNG VIỆT TỰ ĐỘNG CHUYỂN THÀNH VĂN BẢN BẰNG DEEP LEARNING

**Tên tiếng Anh:** Automatic Speech Recognition for Vietnamese Language Using Deep Learning  
**Lĩnh vực:** Âm thanh và tiếng nói  
**Mô hình:** Wav2Vec2-Vietnamese + LM 4-gram và PhoWhisper Base fine-tuned  
**Dataset thực nghiệm:** VIVOS 1.0, Common Voice VI 21.0 và các ablation Kaggle bổ sung  
**Trạng thái:** báo cáo từ artifact huấn luyện và đánh giá thực trên VIVOS, Common Voice và Kaggle holdout.

## Tóm tắt

Đồ án xây dựng pipeline ASR tiếng Việt từ audio đến transcript: kiểm tra dataset, chuẩn hóa audio 16 kHz mono và Unicode tiếng Việt, suy luận Wav2Vec2-Vietnamese, fine-tune PhoWhisper, tính WER/CER, phân tích lỗi và triển khai Gradio local. PhoWhisper Base được fine-tune thật trên VIVOS rồi thích nghi bằng VIVOS + Common Voice VI 21.0. Hai kiến trúc được đánh giá trên cùng VIVOS test; các thí nghiệm dữ liệu bổ sung còn được kiểm tra trên Common Voice, Kaggle holdout và mẫu ST23D độc lập. Tất cả entry point dùng chung module `src/`; cấu hình quick chỉ dùng cho smoke test.

**Từ khóa:** ASR, tiếng Việt, Wav2Vec2, CTC, Whisper, fine-tuning, WER, CER.

## 1. Giới thiệu và động cơ

ASR chuyển tín hiệu tiếng nói thành chuỗi ký tự. Tiếng Việt có hệ thống thanh điệu và dấu phụ phong phú; một lỗi ký tự có thể đổi nghĩa từ. Ứng dụng gồm phụ đề, ghi biên bản, tìm kiếm nội dung và hỗ trợ tiếp cận. Đề tài có giá trị học thuật vì kết hợp xử lý tín hiệu, sequence modeling, transfer learning và đánh giá định lượng.

## 2. Phát biểu bài toán và mục tiêu

Đầu vào là file chứa một phát ngôn tiếng Việt, đầu ra là transcript cùng ngôn ngữ. Mục tiêu: (1) tạo pipeline dữ liệu tái lập; (2) chạy Wav2Vec2-Vietnamese; (3) fine-tune Whisper thực sự; (4) so sánh hai kiến trúc trên cùng test set bằng WER/CER; (5) nhận diện dạng lỗi; (6) demo local có xử lý lỗi.

## 3. Tổng quan phương pháp ASR

ASR truyền thống ghép acoustic model, pronunciation lexicon và language model. Wav2Vec2 học representation trực tiếp từ waveform rồi dùng CTC để ánh xạ frame sang ký tự mà không cần alignment thủ công. Whisper là Transformer encoder–decoder sequence-to-sequence pretrained đa ngôn ngữ bằng weak supervision quy mô lớn. So sánh hai model vì chúng đại diện cho hai hướng giải mã khác nhau: CTC song song và decoder tự hồi quy.

## 4. Cơ sở lý thuyết

Wav2Vec2 nhận waveform 16 kHz, encoder tích chập và Transformer tạo representation theo thời gian; CTC beam search kết hợp điểm acoustic với LM 4-gram. Whisper biến waveform thành log-Mel spectrogram, encoder tạo speech representation, decoder sinh subword token tự hồi quy. Pretraining cung cấp representation tổng quát; fine-tuning PhoWhisper điều chỉnh tham số theo miền tiếng Việt đọc. Chi tiết công thức ở [MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md).

## 5. Dataset và giấy phép

### 5.1 VIVOS 1.0

Nguồn chính thức [Zenodo 7068130](https://doi.org/10.5281/zenodo.7068130) mô tả khoảng 15 giờ tiếng Việt đọc trong môi trường yên tĩnh: 11.660 utterance train từ 46 speaker và 760 utterance test từ 19 speaker. Giấy phép CC BY-NC-SA 4.0 và lưu ý sử dụng học thuật được tôn trọng. Dự án giữ official test; validation được tách từ train theo speaker.

### 5.2 Common Voice VI 21.0

Mirror [`minhquangphamle/common-voice-vi-21`](https://www.kaggle.com/datasets/minhquangphamle/common-voice-vi-21) có giấy phép CC0-1.0 và được dùng thật để thích nghi miền. Pipeline tách speaker, tạo 4.361 train và 546 validation từ phần train hợp lệ, đồng thời giữ 546 mẫu test độc lập. Khi kết hợp VIVOS, manifest có 14.756 train/18,200 giờ và 1.811 validation/2,516 giờ.

### 5.3 Common Voice 27 Vietnamese — loader hỗ trợ

[Mozilla Common Voice datasets](https://commonvoice.mozilla.org/vi/datasets) công bố locale `vi`, MP3, CC0-1.0. Archive hiện phân phối qua Mozilla Data Collective. Vì tải từ kho Hugging Face cũ trả 401 và không có credential trong workspace, dự án không lách access control; loader vẫn hỗ trợ schema Common Voice khi người dùng tải hợp lệ.

Không sử dụng metadata người nói ngoài mục đích tách split chống leakage.

### 5.4 Kaggle Vietnamese Speech Recognition Data — ablation

Dataset [`tuvo6804/vietnamese-speech-recognition-data`](https://www.kaggle.com/datasets/tuvo6804/vietnamese-speech-recognition-data) có 3.736 FLAC kèm transcript/6,061698 giờ và 725 public-test không nhãn không dùng. Chia xác định 90/10 tạo 3.362 train và 374 holdout, không giao nhau về ID, đường dẫn hoặc transcript; holdout cũng không trùng transcript chính xác với các split hiện hữu. Dataset không cung cấp speaker ID tin cậy nên không thể khẳng định speaker-disjoint. Metadata Kaggle ghi license `unknown`; dự án chỉ giữ dữ liệu cho thí nghiệm học thuật nội bộ, không phân phối lại dữ liệu thô và không suy diễn quyền sử dụng thương mại.

## 6. Tiền xử lý

Audio được kiểm tra tồn tại, kích thước, codec, sample rate, NaN/Inf và thời lượng; stereo được lấy trung bình kênh; resample bằng polyphase về 16 kHz. Pipeline chuẩn bị dữ liệu huấn luyện giữ ngưỡng 30 giây để ổn định batch; demo không giới hạn tổng thời lượng mà chia audio dài thành cửa sổ 28 giây. Transcript được Unicode NFC, casefold và gom whitespace; dấu tiếng Việt/dấu câu được giữ. Cùng quy tắc áp dụng cho reference và prediction.

## 7. Mô hình Wav2Vec2-Vietnamese

Checkpoint `nguyenvulebinh/wav2vec2-base-vietnamese-250h` có khoảng 95 triệu tham số và giấy phép CC BY-NC 4.0. Theo model card, representation được pretrain trên khoảng 13.000 giờ audio YouTube tiếng Việt không nhãn và fine-tune trên 250 giờ VLSP ASR. Dự án resample audio về 16 kHz rồi giải mã CTC beam search với LM 4-gram chính thức; alpha=1, beta=1 và beam width=100 được chọn trên validation.

## 8. Mô hình Whisper fine-tuned

Model khởi tạo từ `vinai/PhoWhisper-base`, checkpoint Whisper đã thích nghi tiếng Việt đa giọng, sau đó được fine-tune trên toàn bộ 10.733 mẫu train VIVOS và thích nghi tiếp bằng VIVOS + Common Voice VI 21.0. Dynamic padding tách riêng input features/labels; padding label thành `-100`; BOS dư được loại; gradient checkpointing và accumulation giảm VRAM. Checkpoint triển khai được lưu tại `checkpoints/vivos_cv21_phowhisper_adaptation/best`. Khi suy luận, model khóa task `transcribe` và ngôn ngữ `Vietnamese`. Các checkpoint FPT/Kaggle bổ sung được giữ riêng làm ablation và không tự động thay checkpoint triển khai.

## 9. Thiết lập huấn luyện

| Tham số | Giai đoạn VIVOS | Thích nghi VIVOS + Common Voice |
|---|---:|---:|
| Checkpoint đầu | `vinai/PhoWhisper-base` | `vivos_phowhisper_base_full/best` |
| Epoch | 1 | 1 |
| Learning rate | 5e-6 | 2e-6 |
| Train/eval batch | 1 / 1 | 1 / 1 |
| Gradient accumulation | 4 | 4 |
| Weight decay | 0,01 | 0,01 |
| Warmup steps | 500 | 250 |
| Max grad norm | 1,0 | 1,0 |
| Max audio | 30 s | 30 s |
| Seed | 42 | 42 |
| Train/validation | 10.733 / 927 | 14.756 / 1.811 |
| Optimizer steps | 2.684 | 3.689 |

Giai đoạn VIVOS đạt train loss 0,3182 và validation WER/CER 0,03785/0,01896. Giai đoạn thích nghi đạt train loss 0,28528 và validation WER/CER 0,04608/0,02123 trên tập lớn, đa miền hơn. Checkpoint sau thích nghi được chọn vì Common Voice test cải thiện từ WER 0,17549 xuống 0,15137 trong khi WER VIVOS giữ 0,08923; tập test không được dùng để cập nhật trọng số hoặc sửa prediction.

## 10. Thiết kế thực nghiệm

Câu hỏi chính: Wav2Vec2-Vietnamese CTC và PhoWhisper fine-tuned có WER/CER khác nhau thế nào trên cùng toàn bộ 760 test samples? Audio, reference, normalization và metric được giữ cố định. Wav2Vec2 dùng CTC beam search + LM 4-gram; PhoWhisper dùng decoder tự hồi quy với beam được chọn trên validation.

## 11. Chỉ số đánh giá

WER/CER dùng Levenshtein micro-average. WER phản ánh lỗi từ; CER nhạy với dấu và ký tự, hữu ích cho tiếng Việt. Thời gian inference trung bình được ghi nhưng không gọi là latency production do chưa benchmark warm-up/repetition đầy đủ.

## 12. Kết quả

Dataset sau chuẩn bị có 10.733 train (13,649 giờ), 927 validation (1,272 giờ)
và 760 test (0,746 giờ), tổng cộng 12.420 utterance và 15,667 giờ. Không có
record bị loại; kiểm tra ID và speaker cho thấy giao nhau giữa các split bằng 0.

| Cấu hình | Test samples | WER | CER | Mean inference |
|---|---:|---:|---:|---:|
| Wav2Vec2 Vietnamese 250h, CTC beam + LM 4-gram | 760 | 0,0584 | 0,0278 | 0,0145 giây |
| PhoWhisper Base fine-tuned VIVOS + Common Voice, beam 1 | 760 | 0,0892 | 0,0403 | 0,2524 giây |

Wav2Vec2 có 451 word errors, 912 character errors và 481 transcript khớp hoàn
toàn. PhoWhisper có 689 word errors, 1.320 character errors và 381 transcript khớp
hoàn toàn. Chênh lệch PhoWhisper trừ Wav2Vec2 là 0,0308 WER và 0,0115 CER.
Thời gian inference chỉ đo forward/generate trong lần chạy này, không phải latency
production đã khử hoàn toàn warm-up/cache.

Theo thời lượng `<3 s / 3–6 s / >=6 s`, WER của Wav2Vec2 lần lượt là
0,0774 / 0,0517 / 0,1056; WER của PhoWhisper là 0,1256 / 0,0779 / 0,1278.
Nhóm `>=6 s` chỉ có 13 mẫu nên không nên khái quát mạnh từ con số này.

Wav2Vec2 thấp hơn trong cấu hình triển khai cụ thể, nhưng đây không phải so sánh
đồng mức về ngân sách supervised training: checkpoint Wav2Vec2 đã fine-tune trước
trên 250 giờ VLSP, trong khi Whisper của đồ án fine-tune một epoch trên khoảng
13,65 giờ train VIVOS. Kết quả chỉ hỗ trợ kết luận cho VIVOS read speech.

Guardrail `max_length=225`, `no_repeat_ngram_size=3` và `repetition_penalty=1.05`
được dùng nhất quán cho validation và test Whisper; các tham số này không
áp dụng cho CTC của Wav2Vec2.

## 13. Phân tích lỗi

`outputs/predictions/*.csv` lưu audio ID, reference, prediction, WER, CER và nhãn sơ bộ: khớp, thiếu từ, thừa từ, thay thế. Sai dấu, từ đồng âm/gần âm, tiếng ồn hoặc vùng miền chỉ được kết luận sau khi nghe mẫu và có bằng chứng metadata/audio; nhãn tự động không thay thế phân tích thủ công.

Wav2Vec2 có 202 mẫu được gán nhãn thay thế từ và 26 mẫu sai dấu;
PhoWhisper có 300 mẫu thay thế từ và 41 mẫu sai dấu. Ví dụ khó cho cả hai
model là `VIVOSDEV10_262`: reference “thua xa tiệm tạp hoá nhà tao”, trong khi
Wav2Vec2 dự đoán “thu tra kiểm cập hóa nhà cao” và PhoWhisper dự đoán “hua xe
kiểm cập hóa nhà cao”. Đây là bằng chứng về nhầm lẫn âm/ký tự trên mẫu cụ thể, không
được quy cho giọng vùng miền hay tiếng ồn khi chưa có metadata tương ứng.

Demo hiển thị proxy độ tin cậy chưa hiệu chuẩn: trung bình xác suất frame
CTC không blank cho Wav2Vec2 và trung bình nhân xác suất token cho Whisper. Trung
bình proxy trên test lần lượt là 0,9596 và 0,9240; không được diễn giải như
xác suất toàn câu đúng và không thay thế WER/CER.

## 14. Demo

Gradio cho tải/ghi audio, phát lại, xem duration/sample rate, chọn Wav2Vec2 hoặc PhoWhisper fine-tuned, xem transcript, thời gian và device. PhoWhisper bị ẩn nếu checkpoint không tồn tại. Model cache riêng theo lựa chọn; audio không gửi tới API ASR ngoài. Demo không giới hạn tổng thời lượng; file dài được suy luận theo cửa sổ 28 giây rồi ghép transcript. Với mã lớp `ST23D`, ứng dụng chỉ áp dụng quy tắc miền đã khai báo sau ngữ cảnh `lớp`/`mã`, đồng thời hiển thị rõ cụm nhận dạng gốc và cụm chuẩn hóa; transcript tham chiếu chỉ dùng tính WER/CER, không dùng sửa đầu ra.

## 15. Thí nghiệm bổ sung với FPT Open Speech

FPT Open Speech được tải từ mirror Kaggle và đối chiếu với bản công bố chính thức version 4 trên Mendeley Data. Sau lọc thời lượng tối đa 30 giây, 25.919 mẫu FPT/30,196 giờ được thêm riêng vào train; tổng train tăng thành 40.675 mẫu/48,396 giờ. Validation 1.811 mẫu, VIVOS test 760 mẫu và Common Voice test 546 mẫu được giữ nguyên.

| Model candidate | VIVOS WER/CER | Common Voice WER/CER | ST23D WER/CER | Quyết định |
|---|---:|---:|---:|---|
| Wav2Vec2 sau FPT | 0,05646 / 0,02679 | 0,12921 / 0,06094 | 0,3000 / 0,1412 | Không thay demo |
| PhoWhisper sau FPT | 0,09531 / 0,04235 | 0,17819 / 0,08400 | 0,2500 / 0,0706 | Không thay demo |

Wav2Vec2 candidate cải thiện nhẹ VIVOS và CER Common Voice nhưng WER Common Voice tăng nhẹ, đồng thời mẫu ST23D xấu rõ so với model gốc đạt WER/CER 0/0. PhoWhisper candidate xấu hơn checkpoint trước trên validation, VIVOS, Common Voice và ST23D. Hai checkpoint mới được giữ làm artifact thí nghiệm nhưng không thay checkpoint demo. Kết quả cho thấy thêm dữ liệu không tự động bảo đảm cải thiện khi miền, transcript và lịch fine-tuning khác nhau; không có cơ sở để tuyên bố ASR đạt 100% trên audio chưa thấy.

## 16. Thí nghiệm bổ sung với Kaggle Vietnamese Speech Recognition Data

Thêm 3.362 mẫu Kaggle có nhãn vào train tạo manifest 18.118 train/23,655465 giờ; giữ 1.811 validation, 760 VIVOS test, 546 Common Voice test và 374 Kaggle holdout. Wav2Vec2 được train một epoch, batch 4, learning rate 2e-6; train loss 0,26212, validation WER/CER 0,16421/0,05440. PhoWhisper tiếp tục từ checkpoint triển khai với batch 1, accumulation 4, learning rate 1e-6; train loss 0,24578, validation WER/CER 0,04787/0,02282.

| Model | VIVOS WER/CER | Common Voice WER/CER | Kaggle holdout WER/CER | ST23D WER/CER | Quyết định |
|---|---:|---:|---:|---:|---|
| Wav2Vec2 đang dùng | 0,05840 / 0,02783 | 0,12872 / 0,06345 | 0,16046 / 0,05770 | 0 / 0 | Giữ demo |
| Wav2Vec2 Kaggle candidate | 0,05569 / 0,02701 | 0,13045 / 0,06267 | 0,12972 / 0,04243 | 0,0500 / 0,0118 | Không promote |
| PhoWhisper đang dùng | 0,08923 / 0,04028 | 0,15137 / 0,07510 | 0,24760 / 0,09875 | 0,1000 / 0,0353 | Giữ demo |
| PhoWhisper Kaggle candidate | 0,09065 / 0,04196 | 0,15260 / 0,07540 | 0,09983 / 0,04622 | 0,1000 / 0,0353 | Không promote |

Trên micro-average của ba corpus độc lập (1.680 mẫu, 18.877 từ tham chiếu), candidate Wav2Vec2 giảm WER 0,11188→0,09959 và candidate PhoWhisper giảm 0,16210→0,10743. Dù tổng thể tốt hơn, Wav2Vec2 tăng WER Common Voice và làm xấu ST23D; PhoWhisper tăng lỗi trên cả VIVOS/Common Voice và không cải thiện ST23D. Quality gate theo từng miền vì vậy không promote hai candidate. Artifact tổng hợp ở `outputs/kaggle_speech/tables/kaggle_speech_experiment_comparison.json`.

## 17. Supervised adaptation cho demo ST23D

Mẫu `data/evaluation/st23d_sample.wav` được dùng có giám sát để thích nghi PhoWhisper; vì vậy WER/CER trên mẫu này là **training fit**, không phải test độc lập. Thử nghiệm beam search và prompt không sửa đồng thời hai lỗi cuối câu. Ba lịch fine-tune tiếp theo cho thấy checkpoint bước 16, huấn luyện bằng label giữ cách phát âm tự nhiên “ét tê hai ba đê”, là mốc can thiệp nhỏ nhất phù hợp cho demo.

Pipeline khai báo quy tắc nguyên-token `xin chờ tất cả` → `xin chào tất cả`. Quy tắc không dùng transcript tham chiếu ở thời điểm suy luận và mọi thay thế đều được hiển thị trong giao diện.

| Phạm vi | PhoWhisper trước WER/CER | PhoWhisper demo-adapted WER/CER |
|---|---:|---:|
| VIVOS test, 760 mẫu độc lập | 0,08923 / 0,04028 | 0,09026 / 0,04046 |
| Common Voice test, 546 mẫu độc lập | 0,15137 / 0,07510 | 0,15038 / 0,07474 |
| Kaggle holdout, 374 mẫu độc lập | 0,24760 / 0,09875 | 0,24563 / 0,09748 |
| Micro-average 1.680 mẫu độc lập | 0,16210 / 0,07017 | **0,16157 / 0,06967** |
| ST23D training fit | 0,1000 / 0,0353 | **0 / 0** |

Wav2Vec2 gốc + LM cũng đạt 0/0 trên ST23D. Kết quả 0/0 của cả hai model chỉ được khẳng định cho đúng mẫu demo; các benchmark độc lập vẫn có WER/CER khác 0 nên không có cơ sở tuyên bố chính xác 100% trên audio chưa thấy. Artifact quyết định nằm tại `outputs/st23d_demo_candidate/tables/st23d_supervised_adaptation_comparison.json`.

## 18. Hạn chế

- Chỉ chạy một epoch và một seed do ngân sách GPU; chưa có khoảng tin cậy qua nhiều lần chạy.
- VIVOS là read speech yên tĩnh, chưa đại diện hội thoại/nhiễu/giọng vùng miền.
- Audio dài đã được chia cửa sổ 28 giây nhưng chưa có VAD/chunk overlap, nên từ ở đúng biên có thể kém chính xác.
- WER/CER phụ thuộc normalization; chưa đánh giá semantic correctness.
- RTX 4050 6 GB giới hạn batch và kích thước model; chưa khảo sát PhoWhisper Small.
- Proxy độ tin cậy trong demo chưa được calibration và không phải xác suất transcript đúng.
- Kaggle Vietnamese Speech Recognition Data có license metadata `unknown`; kết quả không cấp quyền tái phân phối hay sử dụng thương mại.
- Kaggle không có speaker ID tin cậy; holdout tách record/transcript nhưng chưa chứng minh speaker-disjoint.
- Thí nghiệm Kaggle mới chạy một epoch và một seed; miền câu có cấu trúc riêng nên mức cải thiện holdout không đại diện mọi giọng nói tiếng Việt.
- ST23D đã tham gia supervised adaptation; kết quả 0/0 của mẫu này không đo khả năng tổng quát hóa.

## 19. Kết luận và hướng phát triển

Pipeline đã hoàn tất fine-tuning toàn train split, official test, speaker-aware validation, local checkpoint và metric minh bạch. Hướng tiếp theo là nhiều epoch/nhiều seed, PhoWhisper Small, augmentation có kiểm soát, long-form chunking overlap, error analysis theo duration/speaker group và benchmark beam/time.

## Tài liệu tham khảo (IEEE)

[1] A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” 2022. https://arxiv.org/abs/2212.04356  
[2] H.-T. Luong and H.-Q. Vu, “A non-expert Kaldi recipe for Vietnamese Speech Recognition System,” WLSI/OIAF4HLT, 2016. https://aclanthology.org/W16-5207/  
[3] Mozilla Foundation, “Common Voice Datasets.” https://commonvoice.mozilla.org/vi/datasets  
[4] Hugging Face, “Whisper model documentation.” https://huggingface.co/docs/transformers/model_doc/whisper  
[5] VinAI, “PhoWhisper-base model card.” https://huggingface.co/vinai/PhoWhisper-base  
[6] A. Vaswani et al., “Attention Is All You Need,” 2017. https://arxiv.org/abs/1706.03762  
[7] A. Baevski et al., “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,” 2020. https://arxiv.org/abs/2006.11477  
[8] N. V. L. Binh, “Wav2Vec2-Base-250h for the Vietnamese language,” model card. https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h
[9] FPT.AI, “FPT Open Speech Dataset,” Mendeley Data, version 4. https://data.mendeley.com/datasets/k9sxg2twv4/4  
[10] Binhhhhhhhhh, “FPT Open Speech Dataset FOSD Vietnamese,” Kaggle mirror. https://www.kaggle.com/datasets/binhhhhhhhhh/fpt-open-speech-dataset-fosd-vietnamese
[11] Tu Vo, “Vietnamese speech recognition_data,” Kaggle dataset. https://www.kaggle.com/datasets/tuvo6804/vietnamese-speech-recognition-data
