# Cơ sở toán học cho Vietnamese ASR với Wav2Vec2 và Whisper

## 1. Biểu diễn audio

Với tín hiệu rời rạc (x[n]) ở 16 kHz, Short-Time Fourier Transform trên frame (m) và bin (k):

\[
X(m,k)=\sum_{n=0}^{N-1}x[n+mH]w[n]e^{-j2\pi kn/N}.
\]

Năng lượng phổ được chiếu qua ngân lọc Mel (H_b[k]):

\[
M(m,b)=\sum_k |X(m,k)|^2H_b[k], \qquad L(m,b)=\log(M(m,b)+\epsilon).
\]

Whisper dùng log-Mel spectrogram 80 bins từ audio mono 16 kHz. Wav2Vec2 nhận
waveform 16 kHz trực tiếp qua feature encoder tích chập. Processor của từng
checkpoint thực hiện phép biến đổi tương ứng; dự án không tự thay đổi tốc độ hay
cao độ.

## 2. Wav2Vec2 và Connectionist Temporal Classification

Wav2Vec2 dùng convolutional feature encoder để biến waveform thành dãy vector
ẩn, sau đó Transformer contextual encoder mô hình hóa ngữ cảnh hai chiều. Với
mỗi frame (t), CTC head cho phân phối trên vocabulary cộng blank:

\[
P(\pi\mid x)=\prod_{t=1}^{T}P(\pi_t\mid x),\qquad
P(y\mid x)=\sum_{\pi\in\mathcal{B}^{-1}(y)}P(\pi\mid x).
\]

Ánh xạ \(\mathcal{B}\) gom token lặp liên tiếp và bỏ blank. Trong đồ án, beam
search giữ nhiều prefix CTC rồi kết hợp log-score acoustic với LM 4-gram:

\[
S(y\mid x)=\log P_{CTC}(y\mid x)+\alpha\log P_{LM}(y)+\beta\,|y|.
\]

Các hệ số \(\alpha=1\), \(\beta=1\) và beam width 100 được chọn trên validation,
không dùng test để điều chỉnh.

## 3. Encoder–decoder Transformer của Whisper

Encoder biến dãy feature thành hidden states. Với ma trận truy vấn, khóa, giá trị:

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
\]

Multi-head attention học nhiều quan hệ thời gian/âm học song song. Decoder tự hồi quy ước lượng:

\[
P(y\mid x)=\prod_{t=1}^{T}P(y_t\mid y_{<t},x).
\]

Whisper đưa token ngôn ngữ và task vào prompt decoder; dự án khóa `language=Vietnamese`, `task=transcribe` để tránh dịch sang tiếng Anh.

## 4. Teacher forcing và loss của Whisper

Khi train, decoder nhận token đúng trước đó (teacher forcing). Cross-entropy có mask padding:

\[
\mathcal{L}=-\frac{1}{|\Omega|}\sum_{t\in\Omega}\log P_\theta(y_t^*\mid y_{<t}^*,x),
\]

với \(\Omega\) là các vị trí label thật. Collator đặt padding label thành `-100`, giá trị PyTorch bỏ qua khi tính loss.

## 5. Tối ưu Whisper fine-tuned

AdamW tách weight decay khỏi moment update. Learning rate của thí nghiệm
PhoWhisper Base đầy đủ là \(5\times10^{-6}\) để giảm nguy cơ phá hỏng
representation pretrained. Gradient accumulation 4 bước mô phỏng batch hiệu
dụng 4 khi batch vật lý là 1. Gradient clipping giới hạn norm ở 1.0.

## 6. Decoding

Với Wav2Vec2-CTC, dự án dùng beam search + LM 4-gram thay cho greedy. Với
Whisper tự hồi quy, greedy chọn token xác suất cao nhất ở mỗi bước; beam search
giữ \(B\) giả thuyết có log-probability tốt nhất. Beam lớn có thể tăng độ chính
xác nhưng chậm hơn và không bảo đảm WER giảm; vì vậy beam được chọn trên
validation trước khi báo cáo một lần trên test.

## 7. WER và CER

Với substitution (S), deletion (D), insertion (I), số đơn vị tham chiếu (N):

\[
WER=\frac{S_w+D_w+I_w}{N_w},\qquad CER=\frac{S_c+D_c+I_c}{N_c}.
\]

WER có thể lớn hơn 1 khi insertion nhiều hơn số từ tham chiếu. Dự án chuẩn hóa Unicode NFC, chữ thường và khoảng trắng đồng nhất cho reference/prediction; giữ dấu tiếng Việt và dấu câu.

## 8. Regularization và overfitting

Validation độc lập giúp phát hiện khi train loss giảm nhưng WER tăng. Weight decay, checkpoint tốt nhất, early stopping và giới hạn epoch là các cơ chế kiểm soát. Test chỉ dùng một lần để báo cáo, không dùng chọn learning rate hoặc epoch.

## 9. ASR khác TTS

ASR học ánh xạ audio \(x\rightarrow y\) là văn bản. Text-to-Speech học ánh xạ văn bản \(y\rightarrow x\) và phải mô hình hóa giọng, prosody, waveform. Wav2Vec2 và Whisper trong đồ án đều là ASR, không sinh audio.
