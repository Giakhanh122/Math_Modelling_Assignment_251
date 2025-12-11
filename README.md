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


#### **📋 requirements.txt**
- **Mục đích**: Liệt kê tất cả các thư viện Python cần thiết
- **Cách dùng**: `pip install -r requirements.txt` để cài đặt tự động





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
**🐍 run.py**
- **Vai trò**: File chạy chương trình
- **Chức năng chính**: Thực hiện chạy cả 5 tác vụ.
  

#### **📁 test_pnml_files/** - *Thư mục chứa dữ liệu testing mạng Petri*
Sơ đồ mô tả input:

- Testcase 1 (medium_deadlock.pnml):

<img width="1035" height="412" alt="image" src="https://github.com/user-attachments/assets/e8d0ca33-32e3-44c6-9dfd-ff9ec5f31f54" />

- Testcase 2 (medium_no_deadlock.pnml):

<img width="945" height="505" alt="image" src="https://github.com/user-attachments/assets/0dffcb3f-1fff-4b9f-9c29-042ca6deb45d" />



## Cài đặt
1. Clone repository về máy
  ``` bash
  git clone https://github.com/Giakhanh122/Math_Modelling_Assignment_251
  ```
2. Vào thư mục repo
  ``` bash
  cd .\Math_Modelling_Assignment_251\
  ```
3. Chạy setup môi trường python ảo để thông dịch Python
   - **Window**
   ```bash
   .\setup\setup.bat
   ```
   - **macOS / Linux**
   ```bash
   source ./setup/setup.sh
   ```
3. Chon testcase bằng cách thay đổi đối số của hàm run trong file run.py
4. Chạy file Python
   ```bash
   python ./source/run.py
   ```
