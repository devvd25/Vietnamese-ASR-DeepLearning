# Câu hỏi và trả lời bảo vệ

1. **ASR là gì?** ASR ánh xạ waveform tiếng nói thành chuỗi văn bản cùng nội dung.
2. **Vì sao chọn đề tài?** Bài toán có ứng dụng thực tế và kết hợp xử lý audio, Transformer, transfer learning, đánh giá và triển khai.
3. **Whisper hoạt động tổng quát thế nào?** Audio thành log-Mel, encoder mã hóa, decoder tự hồi quy sinh subword token theo prompt language/task.
4. **Transformer là gì?** Kiến trúc dùng attention để mô hình hóa quan hệ giữa các vị trí, không cần recurrence tuần tự như RNN.
5. **Attention là gì?** Cơ chế tính trọng số tương quan query–key rồi tổng hợp value; công thức có hệ số chuẩn hóa \(1/\sqrt{d_k}\).
6. **Encoder–decoder là gì?** Encoder tạo representation audio; decoder điều kiện trên representation và token trước để sinh transcript.
7. **Tokenizer hoạt động thế nào?** Nó ánh xạ văn bản sang subword token; subword giúp xử lý từ hiếm và từ ghép.
8. **Fine-tuning là gì?** Tiếp tục cập nhật weights pretrained bằng dữ liệu có nhãn của task/miền đích.
9. **Vì sao cần pretrained model?** Corpus môn học nhỏ; pretraining cho representation và khả năng đa ngôn ngữ tốt hơn khởi tạo ngẫu nhiên.
10. **Dataset gồm gì?** VIVOS có waveform WAV, prompts transcript và speaker ID ẩn danh trong cấu trúc file; dự án dùng train/test chính thức.
11. **Vì sao fallback VIVOS?** Common Voice hiện yêu cầu kênh Mozilla Data Collective và repo cũ trả 401; VIVOS có nguồn Zenodo chính thức, giấy phép học thuật rõ.
12. **Vì sao chia train/validation/test?** Train học weights, validation chọn cấu hình/checkpoint, test ước lượng cuối độc lập.
13. **Vì sao tránh data leakage?** Cùng speaker hoặc utterance ở train/test làm metric lạc quan không phản ánh khả năng tổng quát hóa.
14. **WER là gì?** Tổng substitution, deletion, insertion ở mức từ chia số từ reference.
15. **CER là gì?** Công thức tương tự ở mức ký tự; đặc biệt hữu ích với dấu tiếng Việt.
16. **S, D, I nghĩa là gì?** Thay thế, xóa và chèn cần thiết để biến prediction thành reference.
17. **Vì sao WER có thể lớn hơn 1?** Số insertion có thể khiến tổng lỗi vượt số từ reference.
18. **Vì sao chuẩn hóa tiếng Việt?** Unicode tổ hợp/dựng sẵn, case và whitespace khác nhau có thể gây edit giả dù hiển thị gần giống.
19. **Có bỏ dấu không?** Không. Dấu mang nghĩa; pipeline dùng NFC và giữ dấu/dấu câu theo một quy ước chung.
20. **Vì sao cùng audio có transcript khác nhau?** Khác quy ước dấu câu, viết số, tên riêng, nghe nhầm hoặc ambiguity ngôn ngữ.
21. **Hai model so sánh khác nhau thế nào?** Wav2Vec2 nhận waveform, dùng CTC beam search + LM 4-gram; PhoWhisper nhận log-Mel và dùng decoder tự hồi quy. PhoWhisper Base được fine-tune trên đủ 10.733 mẫu train VIVOS, còn checkpoint Wav2Vec2 đã fine-tune trước trên 250 giờ VLSP.
22. **Fine-tuning chắc chắn tốt hơn không?** Không. Dữ liệu ít, learning rate sai hoặc overfitting có thể làm WER test tăng; phải đo.
23. **Vì sao dùng GPU?** Ma trận attention và backward có tính song song cao; GPU giảm thời gian đáng kể.
24. **Overfitting là gì?** Model khớp train nhưng validation/test xấu; theo dõi validation, regularization và early stopping để kiểm soát.
25. **Greedy và beam search khác nhau?** Greedy giữ một token tốt nhất; beam giữ nhiều giả thuyết nên chậm hơn và không luôn giảm WER.
26. **Tại sao label padding là -100?** Cross-entropy của PyTorch bỏ qua vị trí -100, tránh padding đóng góp loss.
27. **Tại sao dynamic padding?** Chỉ pad đến chiều dài lớn nhất mỗi batch, tiết kiệm bộ nhớ hơn global padding.
28. **Audio dài hơn 30 giây được xử lý thế nào?** Demo không giới hạn tổng thời lượng; audio được chia tuần tự thành cửa sổ 28 giây để phù hợp Whisper short-form và kiểm soát VRAM, sau đó ghép transcript theo thứ tự.
29. **Hạn chế chính?** Thí nghiệm chỉ có một epoch/một seed, VIVOS là read speech sạch, GPU 6 GB, chưa long-form/VAD và chưa phân tích vùng miền đủ bằng chứng.
30. **Nếu có thêm thời gian sẽ cải tiến gì?** Chạy nhiều epoch/seed, thử PhoWhisper Small, controlled augmentation, beam/time benchmark, chunk overlap và phân tích lỗi theo speaker/domain.
31. **ASR khác TTS?** ASR audio→text; TTS text→audio và mô hình hóa giọng/prosody/waveform.
32. **Demo có dùng API ngoài không?** Không; checkpoint Hugging Face được tải/cached rồi inference local. Audio không gửi cho paid ASR.
33. **Kết quả cuối đại diện phạm vi nào?** Kết quả bao phủ official test VIVOS 760 mẫu, nên đại diện cho read speech sạch của corpus này; không tự động khái quát sang hội thoại hoặc môi trường nhiễu.
34. **Vì sao Wav2Vec2 tốt hơn trong kết quả cuối?** Wav2Vec2 + LM 4-gram đạt WER 0,0584 so với 0,0892 của PhoWhisper Base fine-tuned. Checkpoint Wav2Vec2 đã được fine-tune trên 250 giờ VLSP và dùng LM ngoài, trong khi PhoWhisper của đồ án chỉ fine-tune một epoch trên khoảng 13,65 giờ VIVOS; do đó không được quy toàn bộ chênh lệch cho kiến trúc.
35. **CTC khác decoder của Whisper thế nào?** CTC dự đoán token theo frame rồi gom lặp/bỏ blank; Whisper sinh từng token có điều kiện trên các token trước và representation audio.
