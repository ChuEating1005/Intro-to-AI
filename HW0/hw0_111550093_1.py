import cv2

im = cv2.imread('data/image.png')

with open('data/bounding_box.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    # 移除換行並按空格分割，然後將每個元素轉換為整數
    box = list(map(int, line.strip().split()))
    roi = im[box[1]:box[3],box[0]:box[2]]

    # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

cv2.imwrite('hw0_111550093_1.png', im)
