# Dữ liệu

Không commit audio hoặc metadata có thể nhận dạng người nói vào Git.

## Dataset thực nghiệm trong workspace

Dự án dùng **VIVOS 1.0** làm fallback có thể tải tự động vì Common Voice hiện chuyển sang Mozilla Data Collective. VIVOS gồm khoảng 15 giờ tiếng Việt đọc, được phát hành cho mục đích học thuật theo **CC BY-NC-SA 4.0**.

- Nguồn chính thức: https://doi.org/10.5281/zenodo.7068130
- Archive: `vivos.tar.gz`
- MD5 công bố: `72972a8b14050f3f11ea7c3debabd7af`
- Vị trí sau giải nén: `data/raw/vivos/{train,test}`

```powershell
curl.exe -L "https://zenodo.org/records/7068130/files/vivos.tar.gz?download=1" -o data/raw/vivos.tar.gz
Get-FileHash data/raw/vivos.tar.gz -Algorithm MD5
tar -xzf data/raw/vivos.tar.gz -C data/raw
```

## Common Voice (được loader hỗ trợ)

Mozilla Data Collective liệt kê **Common Voice Scripted Speech 27.0 – Vietnamese**, locale `vi`, MP3, CC0-1.0. Tải từ giao diện chính thức, chấp nhận điều khoản rồi giải nén vào `data/raw/common-voice-27-vi/`. Không dùng URL archive tự đoán.

- https://commonvoice.mozilla.org/vi/datasets
- https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk

Loader tự nhận diện các cột `path`/`audio_path`, `sentence`/`transcript` và `client_id`/`speaker_id`.

## FPT Open Speech (dữ liệu Kaggle bổ sung)

- Mirror Kaggle: https://www.kaggle.com/datasets/binhhhhhhhhh/fpt-open-speech-dataset-fosd-vietnamese
- Nguồn công bố chính thức: https://data.mendeley.com/datasets/k9sxg2twv4/4
- Vị trí: `data/raw/fpt-open-speech/{mp3,transcriptAll.txt}`
- Quy mô công bố: 25.921 phát ngôn, khoảng 30 giờ.

Loader đọc toàn bộ các khoảng thời gian trong mỗi dòng `transcriptAll.txt`, hỗ trợ timestamp dạng số thực/ký pháp khoa học và bỏ các bản MP3 hậu tố trùng lặp. FPT chỉ được ghép vào train; validation, VIVOS test và Common Voice test được giữ độc lập.
