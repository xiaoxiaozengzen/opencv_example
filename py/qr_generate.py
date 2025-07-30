import qrcode
import cv2



if __name__ == "__main__":
  # 生成二维码
  img = qrcode.make('Hello, OpenCV!')
  img.save('/mnt/workspace/cgz_workspace/Exercise/opencv_example/output/qrcode.png')