# Tokenization audit

File này giúp kiểm tra tokenizer đang biến văn bản thành chuỗi token như thế nào.

- Tokenizer/model: `hf-internal-testing/tiny-random-distilbert`
- `max_length`: `64`
- Số dòng train/val/test: `25` / `7` / `8`
- Phân bố nhãn train: `{'1': 13, '0': 12}`

## Độ dài chuỗi token trên mẫu train

- Min: `33`
- Mean: `45.36`
- Median: `46.00`
- P90: `52.20`
- P95: `57.00`
- Max: `59`
- Tỷ lệ mẫu có khả năng bị cắt ngắn: `0.00%`

## Gợi ý đọc kết quả

- Nếu tỷ lệ bị cắt ngắn quá cao, hãy thử tăng `--max_length`, nhưng cần quan sát thời gian chạy và bộ nhớ.
- Nếu `max_length` quá lớn, mô hình có thể chạy chậm hơn nhiều mà metric chưa chắc tăng.
- Khi so sánh mô hình, cần giữ split dữ liệu và metric giống nhau.
