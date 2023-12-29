import cv2
import sys
import numpy as np
import os
import pathlib

def decode_norm(bbox, H, W):
    if bbox and float(bbox[0][0]) > 1:
        print(f"The input bbox is not norm format: {bbox}")
        return bbox
    
    bbox = np.array(bbox)
    # print(f"bbox shape: {bbox.shape}, bbox: {bbox}")
    bbox[:, 0] = bbox[:, 0] * W
    bbox[:, 1] = bbox[:, 1] * H
    bbox[:, 2] = bbox[:, 2] * W
    bbox[:, 3] = bbox[:, 3] * H
    bbox = bbox.astype(np.int16)
    return bbox


def plot_bbox(img, bbox, c='r'):
    if isinstance(bbox, list):
        bbox = np.array(bbox)
    
    for i in range(bbox.shape[0]):
        # 把bbox[i]画到img上, bbox
        color = np.random.randint(0, 255, size=3)
        color = tuple(color.tolist())
        # print(i, color)
        # cv2.rectangle(img, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), color, 2)
        if c == 'r' or c == "red":
            cv2.rectangle(img, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), (0, 255, 0), 2)


def plot_car(img, car, c='r'):
    # car format: [point1, point2, point3, point4]
    # point format: [x, y]
    if isinstance(car, list):
        car = np.array(car)
    
    for i in range(car.shape[0]):
        color = np.random.randint(0, 255, size=3)
        color = tuple(color.tolist())
        
        color = (0, 0, 255) if c == 'r' else (0, 255, 0)

        cv2.line(img, (car[i, 0], car[i, 1]), (car[(i+1) % car.shape[0], 0], car[(i+1) % car.shape[0], 1]), color, 2)
        cv2.circle(img, (car[i, 0], car[i, 1]), 8, color, -1)


def plot_text(img, text, pos, fontScale=2, color=(0, 0,255), thickness=2):
    """ 
    plot text on img
    text: list, [text1, text2, text3, ...]
    pos: list, [pos1, pos2, pos3, ...]
    """
    for t, p in zip(text, pos):
        cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)


def parse_result_of_yolos(model, results):
    """
    parse result of yolos

    results format: {
        'scores': scores,
        'boxes': boxes,
        'labels': labels
    }

    Return: dict, k: obj, v: loc
    """
    res = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        obj = model.config.id2label[label.item()]
        score = round(score.item(), 2)
        res[obj] = box
    return res


def save(img, path='./box.jpg', verbose=1):
    if isinstance(path, pathlib.Path):
        path = path.as_posix()
    
    cv2.imwrite(path, img)
    if verbose >=1:
        print(f"img saved in {path}")


def parse_caption(caption, sep=";"):
    """ parse caption to dict

    Args:
        caption: str, caption

    Return: dict, k: caption, v: bbox
    """
    caption = caption.replace(' ', '')
    caption = caption.split(sep)
    print(f"==caption: {caption}")
    # remove '' in caption
    caption = [i for i in caption if i != '']
    caption = dict([i.split(":") for i in caption])
    caption = {k: eval(v) for k, v in caption.items()}

    return caption


def plot_caption(img, caption):
    """ Plotting caption on img

    caption: dict, k:caption, v:bbox
    """
    for k, v in caption.items():
        cv2.rectangle(img, (v[0], v[1]), (v[2], v[3]), (0, 255, 255), 2)
        cv2.putText(img, k, (v[0], v[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


def parse_bbox(bboxes, sep=";"):
    bboxes = bboxes.strip().split(sep)
    cname_list = []
    box_list = []
    for item in bboxes:
        cname, box = item.split(":")
        cname_list.append(cname)
        box_list.append(eval(box))
    return cname_list, box_list


if __name__ == "__main__":
    img = "/ssd/wphu/Dataset/lavis/eventgpt/images/GACNE-guangzhou-xhthwk-20210717/ch01003_20210717135428.jpg"  # index=6
    img = "/ssd/wphu/Dataset/lavis/eventgpt/images/GACNE-guangzhou-xhthwk-20210717/ch01001_20210717200618.jpg"  # index=60

    img = cv2.imread(img)
    H, W, _ = img.shape

    ######################################### caption begin #################################
    caption = "man wearing a black shirt: [580, 351, 846, 988]; white suv parked on the street: [1429, 58, 1821, 342]; woman wearing black dress: [1080, 390, 1364, 980]; man wearing a white shirt and black pants: [870, 344, 1064, 901]; man wearing a black shirt and khaki shorts: [191, 307, 412, 854]; woman wearing pink shirt: [2229, 78, 2373, 412]; woman wearing a dress: [2058, 104, 2198, 468]; a woman wearing a black skirt and white shirt: [1943, 66, 2052, 395]; the floor is white: [0, 94, 2559, 1418]; the word urinoir on the bottom: [1970, 1345, 2331, 1434]; words on the building: [4, 0, 1172, 83]; black shoes on the womans feet: [1123, 892, 1362, 978]; the man is wearing a black and white shirt: [593, 445, 800, 678]; grey chair at table: [5, 1252, 360, 1433];"
    caption1 ="black car parked on the street: [1393, 290, 2135, 911]; the car is orange and black: [1035, 111, 1518, 415]; black car parked on the left: [3, 581, 450, 1432]; white car parked on the street: [1805, 100, 2190, 427]; the carpet is white: [0, 73, 2544, 1426]; the black chair on the left: [256, 446, 425, 655]; round table with paper on top: [348, 428, 540, 654]; man sitting at table: [711, 314, 949, 712]; woman wearing white shirt: [19, 474, 198, 697]; the black case is round: [970, 585, 1103, 725]; white car on the corner: [1381, 906, 2549, 1431]; words on top of business: [6, 0, 1165, 82]; the chair is black: [726, 508, 941, 741]; blue chair on the floor: [1095, 454, 1255, 684]; the word the on the black box: [1967, 1343, 2329, 1435]; person sitting on a chair: [963, 351, 1118, 683]"
    caption2 = "car : [ 708, 333, 2185, 1362 ]; person : [ 315, 403, 406, 546 ]; dining table : [ 1, 535, 583, 942 ]; chair : [ 460, 575, 630, 801 ]; tv : [ 734, 182, 1832, 390 ]"

    caption = parse_caption(caption1)
    # caption.update(parse_caption(caption2, sep=","))
    ############################################## caption end #################################

    ############################################## bbox begin #################################
    # bbox format: [x1, y1, x2, y2]
    bbox = []
    gt_bbox = []
    car = []
    gt_car = []
    text = []

    # bbox = [[251.86, 862.77, 452.87, 1191.68], [1598.2, 292.75, 1806.12, 483.05], [723.88, 592.21, 900.07, 818.56],
    #         [924.19, 448.47, 2363.9, 1418.52], [535.08, 966.98, 837.31, 1434.07], [951.41, 506.52, 1176.2, 794.15],
    #         [1401.59, 302.67, 1592.41, 541.28]]
    # bbox = [[int(i) for i in j] for j in bbox]

    # bbox = [[1, 388, 910, 1012],[994, 285, 1173, 791],[411, 761, 2317, 1428],[169, 497, 431, 1070],[1775, 316, 1946, 635],[1154, 234, 1329, 752],[1311, 296, 1473, 744],[2267, 488, 2493, 667],[1446, 181, 1570, 574],[0, 11, 2560, 1412],[5, 1, 1145, 74],[1969, 1347, 2333, 1434],[1805, 3, 2550, 220]]
    bbox = [
            [0.001, 0.274, 0.356, 0.755],
            ]
    
    # bboxes = "car: [1372, 952, 2468, 1425]; car: [1406, 298, 2145, 898]; car: [1395, 901, 2553, 1427]; car: [1, 656, 471, 1423]"
    # text, bbox = parse_bbox(bboxes)

    # text = ["person", "car"] * 10
    # text = [str(i) for i in range(len(bbox))]
    txt_pos = [(int(i[0] * W), int(i[1] * H)) for i in bbox]
    # # car = [ [ 2048, 489 ], [ 2176, 604 ], [ 896, 633 ], [ 947, 503 ] ]

    gt_bbox = [[0.243, 0.537, 0.898, 0.994]]
    # gt_bbox = "person:[0.804, 0.117, 0.858, 0.374];person:[0.162, 0.578, 0.255, 0.992];person:[0.232, 0.544, 0.381, 0.99];person:[0.198, 0.292, 0.305, 0.732];person:[0.42, 0.26, 0.502, 0.565];person:[0.918, 0.437, 1.0, 0.969]"
    # gt_text, gt_bbox = parse_bbox(gt_bbox)
    # gt_car = [[0,676],[51,489],[1433,331],[1689,475]]


    if len(np.array(bbox).shape) == 1:
        bbox = [bbox]
    ############################################## bbox end #################################

    ############################################## plot begin #################################
    # Plot pred bbox on img
    if bbox != [[]]:
        bbox = decode_norm(bbox, H, W)
        print(f"bbox: {bbox}")
        plot_bbox(img, bbox, c='r')

    if car:
        plot_car(img, car, c='r')

    ## Plot GT bbox on img
    if gt_bbox:
        gt_bbox = decode_norm(gt_bbox, H, W)
        plot_bbox(img, gt_bbox, c='g')

    if gt_car:
        plot_car(img, gt_car, c='g')

    if text:
        plot_text(img, text, txt_pos)

    # if caption:
    #     plot_caption(img, caption)

    # save img
    save(img)



