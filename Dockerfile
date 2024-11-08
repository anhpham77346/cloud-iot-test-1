# Sử dụng phiên bản Python mong muốn
FROM python:3.10.0

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp yêu cầu vào container và cài đặt các phụ thuộc
COPY requirements.txt .
RUN pip install -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Khởi chạy ứng dụng
CMD ["python", "testing.py"]
