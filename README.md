<div align="center">
<h1> Symbolic and Algebraic Reasoning in Petri Nets </h1>

![Python](https://img.shields.io/badge/python-3.8-blue)
![Status](https://img.shields.io/badge/status-complete-success)
</div>

## Thông tin nhóm
- **Môn học**: Mathematical Modeling (CO2011)
- **Bài tập**: Symbolic and Algebraic Reasoning in Petri Nets
- **Nhóm**: 40
- **Thành viên**:
  - 2411549 - Võ Văn Gia Khánh
  - 2410281 - Nguyễn Tiến Vũ Bảo
  - 2411256 - Phạm Gia Huy
  - 2410851 - Trần Mậu Giàu
  - 2411184 - Lê Anh Huy

## Yêu cầu hệ thống
- Python 3.8+
- Các thư viện cần thiết (xem `requirements.txt`)
- Trình biên dịch C++ (nếu dùng C++), hoặc JDK (nếu dùng Java)
## Cấu trúc mã nguồn
#### **📄 README.md**
- **Mục đích**: Tài liệu hướng dẫn sử dụng project
- **Nội dung**: Cách cài đặt, chạy chương trình, cấu trúc project, giải thích các tính năng

#### **📋 requirements.txt**
- **Mục đích**: Liệt kê tất cả các thư viện Python cần thiết
- **Cách dùng**: `pip install -r requirements.txt` để cài đặt tự động

#### **🖼️ image.png**
- **Mục đích**: Hình ảnh minh họa cho project
- **Nội dung có thể**: Sơ đồ kiến trúc, flowchart, diagram mạng Petri, kết quả phân tích


#### **📁 setup/** - *Thư mục thiết lập môi trường*
**🪟 setup.bat**
- **Hệ điều hành**: Windows
- **Mục đích**: Tự động hóa việc tạo virtual environment và cài đặt dependencies
- **Các bước thực hiện**: 
  1. Tạo môi trường ảo Python
  2. Kích hoạt môi trường
  3. Cài đặt packages từ requirements.txt
  4. Thiết lập các biến môi trường cần thiết

**🐧 setup.sh**
- **Hệ điều hành**: macOS và Linux
- **Mục đích**: Tương tự setup.bat nhưng dùng shell script
- **Đặc điểm**: Có thể bao gồm kiểm tra phiên bản Python, phân quyền thực thi


#### **📁 source/** - *Thư mục mã nguồn chính*
**🐍 source.py**
- **Vai trò**: File chính thực thi chương trình
- **Chức năng chính**:
  - Đọc và phân tích các file PNML từ thư mục `test_pnml_files/`
  - Thực hiện phân tích mạng Petri (deadlock detection, reachability, v.v.)
  - Xuất kết quả phân tích
  - Có thể chứa các hàm xử lý đồ họa/visualization


#### **📁 test_pnml_files/** - *Thư mục chứa dữ liệu testing mạng Petri*

**📊 Phân loại theo kích thước:**
- **📉 small.pnml**: Mạng Petri nhỏ, ít places và transitions
- **📈 medium.pnml**: Mạng Petri trung bình
- **📊 medium_petri_net.pnml**: Mạng Petri trung bình (biến thể)
- **📊 large.pnml**: Mạng Petri lớn, độ phức tạp cao

**⚠️ Phân loại theo tính chất deadlock:**
- **⚠️ deadlock_simple_1.pnml**: Deadlock cơ bản, dễ nhận diện
- **🔄 conflict_deadlock.pnml**: Deadlock do conflict giữa các transitions
- **🔁 loop_reach_deadlock.pnml**: Deadlock từ vòng lặp trong mạng
- **✅ loop_safe_1.pnml**: Vòng lặp an toàn (không gây deadlock)

**🧪 File testing đa năng:**
- **test_file.pnml**: File test tổng hợp nhiều kịch bản

### 🎯 Mục đích sử dụng các file test:

| Loại file | Mục đích testing | Độ phức tạp |
|-----------|------------------|-------------|
| **small.pnml** | Unit test, debug nhanh | Thấp |
| **medium.pnml** | Integration test | Trung bình |
| **large.pnml** | Performance test, stress test | Cao |
| **deadlock_*.pnml** | Test deadlock detection | Đa dạng |
| **loop_*.pnml** | Test cyclic behavior analysis | Đa dạng |
## Cài đặt
1. Clone repository về máy
  ``` bash
  git clone https://github.com/Giakhanh122/Math_Modelling_Assignment_251
  ```
2. Vào thư mục repo
  ``` bash
  cd Math_Modelling_Assignment_251
  ```
4. Chạy setup môi trường python ảo để thông dịch Python
   - **Window**
   ```bash
   .\setup\setup.bat
   ```
   - **macOS / Linux**
   ```bash
   source ./setup/setup.sh
   ```
5. Chạy file Python
   ```bash
   python ./source/run.py
   ```
