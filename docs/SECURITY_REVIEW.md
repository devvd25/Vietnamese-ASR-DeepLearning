# Web Security Review — Gradio ASR Demo

## Phạm vi

- Framework: Python/Gradio 6.30, local monolith.
- Mặc định bind `127.0.0.1`; không authentication, database hoặc API nghiệp vụ.
- Trust boundary: file audio do người dùng upload; transcript có thể là nội dung nhạy cảm.
- Không lưu audio upload vào dataset và không gửi audio tới dịch vụ ASR ngoài.

## OWASP Top 10 matrix

| OWASP | Trạng thái | Mức còn lại | Nhận xét |
|---|---|---:|---|
| A01 Broken Access Control | Không có tài khoản/role | N/A local | Không dùng cho multi-user public nếu chưa thêm auth/reverse proxy. |
| A02 Cryptographic Failures | HTTP localhost | N/A local | Public deployment phải đặt sau HTTPS; không có secret trong source. |
| A03 Injection | Đã kiểm tra | Thấp | Không shell/SQL/template eval từ input; Gradio render output bằng component text/Markdown. |
| A04 Insecure Design | Có guardrails | Trung bình | Giới hạn 100 MB, chia cửa sổ 28 giây và queue concurrency 1; do không giới hạn tổng thời lượng, public share có nguy cơ compute abuse cao hơn. |
| A05 Misconfiguration | Có guardrails | Thấp | Bind localhost mặc định; lỗi ngoài dự kiến không trả stack/path cho UI. Headers public do reverse proxy đảm nhiệm. |
| A06 Vulnerable Components | Chưa SCA | Chưa xác định | Direct dependencies được pin; cần `pip-audit`/OSV trước khi public deploy. |
| A07 Auth Failures | Không có auth | N/A local | Thêm auth/rate limit nếu mở internet. |
| A08 Integrity Failures | Một phần | Thấp | Dataset phải kiểm checksum; model dùng nguồn/checkpoint cố định, weights không commit Git. |
| A09 Logging/Monitoring | Tối thiểu | Thấp local | Không log transcript/audio để bảo vệ riêng tư; chưa có monitoring production. |
| A10 SSRF | Không có URL input/fetch | N/A | Model/dataset URL nằm trong cấu hình do operator kiểm soát. |

## Controls quan sát được

- Allowlist extension và decoder xác minh nội dung thực; từ chối file rỗng/quá 100 MB; audio dài được xử lý tuần tự theo cửa sổ 28 giây.
- Chỉ đọc path upload, không ghép path vào shell command.
- Không dùng paid ASR API, `eval`, `exec`, pickle từ upload, SQL hay URL do người dùng cung cấp.
- Fine-tuned checkpoint thiếu gây lỗi rõ, không fallback giả.
- `.gitignore` loại token, `.env`, dataset và weights.
- Public tunnel chỉ bật khi operator truyền `--share`; tài liệu cảnh báo phạm vi.

## Khuyến nghị trước public deployment

1. Đặt sau reverse proxy HTTPS với authentication, rate limiting, request/body limits và security headers.
2. Chạy SCA trên `requirements.txt`, cập nhật bản vá và tạo lock/hash nếu deploy lâu dài.
3. Tách worker inference, timeout và quota GPU; giám sát lỗi/latency nhưng không log audio/transcript thô.
4. Cấu hình retention/xóa temp uploads và chính sách quyền riêng tư rõ ràng.
