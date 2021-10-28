import cv2
# import imutils
# from imutils import rotate_bound

image = cv2.imread('char_seg/tl25_Image_00762.jpg', 0)
# 获取图像的维度，并计算中心
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)

# 逆时针以图像中心旋转45度
# - (cX,cY): 旋转的中心点坐标
# - 45: 旋转的度数，正度数表示逆时针旋转，而负度数表示顺时针旋转。
# - 1.0：旋转后图像的大小，1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
# OpenCV不会自动为整个旋转图像分配空间，以适应帧。旋转完可能有部分丢失。如果您希望在旋转后使整个图像适合视图，则需要进行优化，使用imutils.rotate_bound.
M = cv2.getRotationMatrix2D((cX, cY), 90-89.25594329833984, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("Rotated by 45 Degrees", rotated)
# cv2.namedWindow("img",0)
# cv2.resizeWindow("img", 800, 800) # 限定显示图像的大小
# cv2.imshow('img', rotated)

img = rotated[30:h-60, 260:w-40]
# cv2.imwrite('char_seg/img.png', img)
# cv2.namedWindow("img1",0)
# cv2.resizeWindow("img1", 800, 800) # 限定显示图像的大小
# cv2.imshow('img1', img)
cv2.waitKey(0)
