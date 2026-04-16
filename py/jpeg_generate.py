import numpy as np
import cv2

def jpeg_generate_from_bgr_raw():
    width = 1920
    height = 1024

    input_bgr_raw_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/frontwide_1920_1024_3.bgr"
    print(f"Reading BGR raw data from: {input_bgr_raw_file}")

    with open(input_bgr_raw_file, "rb") as f:
        bgr_data = np.frombuffer(f.read(), dtype=np.uint8)
        
    # 分离平面
    size = width * height
    r_plane = bgr_data[0:size].reshape((height, width))
    g_plane = bgr_data[size:2*size].reshape((height, width))
    b_plane = bgr_data[2*size:3*size].reshape((height, width))

    img = cv2.merge([b_plane, g_plane, r_plane])
    output_bgr_jpg_file = "/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/frontwide_1920_1024_3.jpg"
    cv2.imwrite(output_bgr_jpg_file, img)
    print(f"Saved JPEG image to: {output_bgr_jpg_file}")
    
if __name__ == "__main__":
    jpeg_generate_from_bgr_raw()