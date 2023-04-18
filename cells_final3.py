import numpy as np
import cv2
# from cv2 import pyrMeanShiftFiltering

# заполнение дыр
def fill_holes (img):
    (h, w) = img.shape

    img_enlarged = np.zeros ((h + 2, w + 2), np.uint8)
    img_enlarged [1:h+1, 1:w+1] = img

    img_enl_not = cv2.bitwise_not (img_enlarged)
    th, im_th = cv2.threshold (img_enl_not, 220, 255, cv2.THRESH_BINARY_INV)

    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv

    result = im_out [1:h-1, 1:w-1]   
    return result

# наложение маски и работа с ней
def image_filter (hsv_img, low_th, high_th):
    mask = cv2.inRange(hsv_img, low_th, high_th)
    mask_filled = fill_holes(mask)
    mask_final = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, np.ones((9, 9)))
    return mask_final

result_file = open("result.csv", 'w')

# массив выборанных картинок
all_selected_cells = [2, 7, 17, 27, 28, 30, 32, 39, 46, 64, 65, 72, 83, 
                      98, 115, 122, 125, 141, 143, 149, 150, 156, 158, 
                      176, 221, 247, 248, 259, 272, 273, 276, 280, 
                      287, 318, 333, 369, 388, 398, 402, 412, 414, 
                      426, 428, 432, 445, 449, 461, 468, 472, 478, 
                      497, 502, 503, 514, 516, 520, 523, 541, 584, 
                      609, 611, 616, 622, 628]
for i in range (len(all_selected_cells)):
    selected_cell = all_selected_cells[i]  # номер картинки
    if selected_cell < 10: file_name = "00"+str(selected_cell)
    elif selected_cell > 99: file_name = str(selected_cell)
    else: file_name = "0"+str(selected_cell)
    img = cv2.imread("19001-19630/ART_19" + file_name + ".jpg")
    imageSegment = cv2.imread("Segmented images/" + file_name + ".jpg")

    # сегментирование изображения
    # spatialRadius = 150
    # colorRadius = 25
    # imageSegment = pyrMeanShiftFiltering(img, spatialRadius, colorRadius)

    # наложение масок
    hsv = cv2.cvtColor(imageSegment, cv2.COLOR_RGB2HSV)
    lth, hth = (0, 193, 0), (179, 255, 255) # граничные значения для фиолетового
    mask = image_filter(hsv, lth, hth)
    lth_1, hth_1 = (0, 0, 134) , (179, 139, 180) # граничные значения для серого
    mask_1 = image_filter(hsv, lth_1, hth_1)
        
    # работа с контурами и вывод результата
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # контуры фиолетовых клеток
    num_cells = len(contours) # кол-во фиолетовых клеток
    for j in range (num_cells):
        result = "ART_19" + file_name + ",Cell," + str(j + 1)
        for k in range(len(contours[j])): result += ',' + str(contours[j][k][0][0]) + ',' + str(contours[j][k][0][1])
        result_file.write(result + '\n')

    contours_1, hierarchy_1 = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # контуры серых клеток
    num_cells_1 = len(contours_1) # кол-во серых клеток
    for j in range (num_cells_1):
        result = "ART_19" + file_name + ",Cell," + str(num_cells + j + 1)
        for k in range(len(contours_1[j])): result += ',' + str(contours_1[j][k][0][0]) + ',' + str(contours_1[j][k][0][1])
        result_file.write(result + '\n')

# иллюстрация результата
# cv2.drawContours(img, contours, -1, (0,0,255), 1, cv2.LINE_AA, hierarchy, 1)
# cv2.drawContours(img, contours_1, -1, (0,0,255), 1, cv2.LINE_AA, hierarchy_1, 1)
# cv2.imshow("input #" + str(selected_cell), img)
# cv2.waitKey(6000)

result_file.close()