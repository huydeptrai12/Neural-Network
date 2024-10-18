import socket

# Thiết lập kết nối
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('27.77.77.203', 5008)  # Địa chỉ IP và cổng của máy nhận
client_socket.connect(server_address)

# Mở file để gửi
file_path = 'file.txt'
with open(file_path, 'rb') as file:
    file_data = file.read()

# Gửi nội dung file
client_socket.send(file_data)

# Đóng kết nối
client_socket.close()