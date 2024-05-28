from PIL import Image
from PIL import ImageDraw
# Hàm để chuyển đổi ảnh sang trắng đen


def chuyen_anh_sang_trang_den(anh):
    return anh.convert('L')

# Hàm để phát hiện trái cây trong ảnh trắng đen


def phat_hien_trai_cay(anh_trang_den):
    pixel = anh_trang_den.load()
    width, height = anh_trang_den.size
    left, top, right, bottom = width, height, 0, 0

    # Tìm vị trí của trái cây
    for x in range(width):
        for y in range(height):
            if pixel[x, y] < 200:  # Ngưỡng để phát hiện trái cây, có thể điều chỉnh
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    # Trả về tọa độ của vùng chứa trái cây
    return left, top, right, bottom

# Hàm để vẽ vùng trái cây đã phát hiện trên ảnh gốc


def ve_khoanh_vung(anh_goc, left, top, right, bottom):
    draw = ImageDraw.Draw(anh_goc)
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)
    anh_goc.show()


# Đường dẫn của ảnh đầu vào
duong_dan_anh = "C:\Users\ADMIN\Desktop\trituenhantao\buoi3\d.jpg"

# Load ảnh đầu vào
anh = Image.open(duong_dan_anh)

# Chuyển đổi ảnh sang trắng đen
anh_trang_den = chuyen_anh_sang_trang_den(anh)

# Phát hiện trái cây trong ảnh trắng đen
left, top, right, bottom = phat_hien_trai_cay(anh_trang_den)

# Vẽ vùng trái cây đã phát hiện lên ảnh gốc
ve_khoanh_vung(anh, left, top, right, bottom)