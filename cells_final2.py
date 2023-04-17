import numpy as np
import cv2

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

# анализ связных компонент
def connected_component_analysis (img, min_area = 0, min_density = 0):
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        density = a / (w * h)
        if(a < min_area) or (density < min_density): 
            img[np.where(labels == i)] = 0

result_file = open("result.csv", 'w')

# параметры для повышения контрастности
clipLimit_value, tileGridSize_value = 7 , 14
clahe = cv2.createCLAHE(clipLimit=clipLimit_value, tileGridSize=(tileGridSize_value,tileGridSize_value))

# выбор картинки
selected_cell = 155 # номер картинки
if selected_cell < 10: file_name = "00"+str(selected_cell)
elif selected_cell > 99: file_name = str(selected_cell)
else: file_name = "0"+str(selected_cell)
img = cv2.imread("19001-19630\ART_19" + file_name + ".jpg")

### Работа с фиолетовыми клетками ###
# повышение контрастности входного изображения
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l2 = clahe.apply(l)
lab = cv2.merge((l2,a,b))
lab_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# наложение маски и работа с ней
lth, hth = (0, 128, 0), (179, 255, 255) # граничные значения для фиолетового
hsv = cv2.cvtColor(lab_img, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, lth, hth)
mask_filled = fill_holes(mask)
mask_eroded = cv2.erode(mask_filled, np.ones((13, 13)))
connected_component_analysis(mask_eroded, min_area = 150)
mask_dilated = cv2.dilate(mask_eroded, np.ones((13, 13)))
    
# работа с контурами и вывод результата
contours, hierarchy = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_cells = len(contours) # кол-во клеток
for j in range (num_cells):
    result = "ART_19" + file_name + ",Cell," + str(j + 1)
    for k in range(len(contours[j])): result += ',' + str(contours[j][k][0][0]) + ',' + str(contours[j][k][0][1])
    result_file.write(result + '\n')
### Конец работы с фиолетовым ###

### Работа с серыми клетками ###
# наложение маски и работа с ней
lth_1, hth_1 = (0, 0, 121) , (179, 70, 188) # серый
hsv_1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # сразу к HSV без изменения контрастности
mask_1 = cv2.inRange(hsv_1, lth_1, hth_1)
mask_filled_1 = fill_holes(mask_1)
mask_eroded_1 = cv2.erode(mask_filled_1, np.ones((13, 13)))
connected_component_analysis(mask_eroded_1, min_area = 150)
mask_dilated_1 = cv2.dilate(mask_eroded_1, np.ones((11, 11)))
    
# работа с контурами и вывод результата
contours_1, hierarchy_1 = cv2.findContours(mask_dilated_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_cells_1 = len(contours_1) # кол-во клеток
for j in range (num_cells_1):
    result = "ART_19" + file_name + ",Cell," + str(num_cells + j + 1)
    for k in range(len(contours_1[j])): result += ',' + str(contours_1[j][k][0][0]) + ',' + str(contours_1[j][k][0][1])
    result_file.write(result + '\n')
### Конец работы с серым ###

# иллюстрация результата
cv2.drawContours(img, contours, -1, (0,0,255), 1, cv2.LINE_AA, hierarchy, 1)
cv2.drawContours(img, contours_1, -1, (0,0,255), 1, cv2.LINE_AA, hierarchy_1, 1)
cv2.imshow("input #" + str(selected_cell), img)
cv2.waitKey(6000)

result_file.close()
