# Ứng dụng mô hình máy học trong dự báo phân loại lỗi log

Phân loại mức độ nghiêm trọng (**Information / Warning / Error**) của Windows Event Log chỉ dựa trên nội dung `Message`, so sánh 3 mô hình máy học truyền thống và 1 mô hình học sâu.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `src/export_events.ps1` | Xuất log Application + System ra `data/events_raw.csv` |
| `src/train.py` | Tiền xử lý, huấn luyện, đánh giá 4 mô hình, vẽ hình |
| `data/events_raw.csv` | Dữ liệu gốc (không đưa lên repo) |
| `outputs/events_clean.csv` | Dữ liệu đã ẩn danh, khử trùng lặp (bản công khai đã bỏ thêm 198 dòng còn chứa tên máy chủ nội bộ, nên có 2.251 thay vì 2.449 dòng) |
| `outputs/results.csv`, `outputs/summary.json` | Bảng kết quả và thống kê |
| `outputs/*.png` | Phân bố lớp, confusion matrix, so sánh mô hình, đường huấn luyện BiLSTM |
| `requirements.txt` | Danh sách thư viện Python |

## Cách chạy lại

```powershell
pip install -r requirements.txt
.\src\export_events.ps1     # bước 1: lấy dữ liệu (bỏ qua được, dùng outputs/events_clean.csv)
python src\train.py         # bước 2: huấn luyện và đánh giá
python src\cv.py            # bước 3: kiểm định chéo 5-fold
```

`data/events_raw.csv` chứa thông tin của máy thu thập nên không được đưa lên repo.

## Quy trình

1. **Thu thập:** `Get-WinEvent` lấy tối đa 200.000 event từ log Application và System.
2. **Gán nhãn** theo cột `Level`: 0, 4 → Information; 3 → Warning; 1, 2 → Error (Critical quá ít nên gộp vào Error).
3. **Làm sạch và ẩn danh:** bỏ Message rỗng; bỏ log của phần mềm nội bộ công ty (chứa dữ liệu nghiệp vụ); thay URL, email, GUID, SID, IP, đường dẫn, hostname, số, mã hex, tên máy và tên người dùng bằng token (`<url>`, `<ip>`, `<num>`…); chuyển chữ thường.
4. **Khử trùng lặp:** cùng một nội dung xuất hiện hàng nghìn lần. Nếu để nguyên, cùng một câu sẽ nằm ở cả tập train và test, làm điểm cao ảo. Mỗi nội dung chỉ giữ 1 dòng, nhãn là nhãn xuất hiện nhiều nhất.
5. **Chia dữ liệu:** 80% train, 20% test, phân tầng theo nhãn (stratified).
6. **Mô hình:**
   - Logistic Regression, Random Forest, Linear SVM trên đặc trưng TF-IDF (unigram + bigram), chọn siêu tham số bằng GridSearchCV 3-fold, tối ưu macro F1, có cân bằng trọng số lớp.
   - BiLSTM (PyTorch): embedding 128 chiều, BiLSTM 128 chiều mỗi hướng, mean + max pooling, cross-entropy có trọng số lớp, early stopping theo macro F1 trên tập validation.
7. **Đánh giá:** accuracy, **macro F1** (chỉ số chính vì dữ liệu mất cân bằng), F1 từng lớp, confusion matrix.
8. **Kiểm định chéo 5-fold** (`src/cv.py`): báo cáo trung bình ± độ lệch chuẩn macro F1, vì tập test một lần chia chỉ có khoảng 30 dòng Warning.
9. **Thí nghiệm phụ:** (a) không khử trùng lặp: macro F1 gần 1.0 do cùng một câu nằm ở cả train và test (rò rỉ dữ liệu); (b) thêm metadata `ProviderName` + `Id`: không cải thiện, vì sau khi khử trùng lặp nội dung Message đã đủ thông tin.

## Kết quả chính

| Mô hình | Macro F1 (1 lần chia) | Macro F1 (5-fold, TB ± ĐLC) |
|---|---|---|
| Logistic Regression | 0.940 | 0.932 ± 0.014 |
| Random Forest | 0.972 | 0.928 ± 0.012 |
| Linear SVM | 0.966 | 0.929 ± 0.016 |
| BiLSTM | 0.930 | 0.908 ± 0.025 |
