import sys
from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# result =  "/home/wphu/chatglm/LAVIS/eventgpt/output/BLIP2/Oracle/20230911033/log.txt"  # result path
# result2 = "/home/wphu/chatglm/LAVIS/eventgpt/output/BLIP2/Oracle/20230914110/log.txt"
# result3 = "/home/wphu/chatglm/LAVIS/eventgpt/output/BLIP2/LlavaStyle/Pretrain_stage2/20231205062/checkpoint_99.pth"

# label1 = 'Loss_raw'
# label2 = 'Loss_clean'
# label3 = 'augment'

# if len(sys.argv) > 1:
#     result = sys.argv[1]

# if len(sys.argv) > 2:
#     result2 = sys.argv[2]

# if len(sys.argv) > 3:
#     result3 = sys.argv[3]

# if len(sys.argv) > 4:
#     label1 = sys.argv[4]

# if len(sys.argv) > 5:
#     label2 = sys.argv[5]

# if len(sys.argv) > 6:
#     label3 = sys.argv[6]


def parse_result(result_path):
    """Read results from txt and plot loss
    """
    if not Path(result_path).exists():
        print(f"{result_path} not exist!")
        exit()
    
    with open(result_path) as f:
        lines = f.readlines()

    losses = []
    for line in lines:
        if 'train_loss' in line:
            line_dict = eval(line)
            loss = line_dict['train_loss']
            losses.append(eval(loss))

    print(f"losses: {losses}")
    return losses


def plot_losses(losses, color='blue', label='Loss'):
    """ Plot losses
    """
    epochs = range(1, len(losses) + 1)

    # 绘制散点图
    # plt.figure(figsize=(10, 6))
    # plt.scatter(epochs, losses, color=color, marker='o', label='Loss')
    # 绘制折线图
    plt.plot(epochs, losses, color=color, marker='o', label=label)
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


def init():
    plt.figure(figsize=(10, 6))

def save():
    # 保存图表为图片文件
    plt.savefig('loss_plot.png')
    

def random_choose_color():
    """
    随机选择颜色
    """
    color_list = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan', 'magenta', 'gold', 'silver', 'lime', 'maroon', 'navy', 'teal', 'olive', 'indigo', 'crimson', 'azure', 'violet', 'wheat', 'beige', 'chocolate', 'coral']
    # 从color_list中随机选择一个返回
    color = random.choice(color_list)
    return color

if __name__ == "__main__":
    # losses = parse_result(result)
    # losses2 = parse_result(result2)
    # losses3 = parse_result(result3)

    init()

    # plot_losses(losses, color='blue', label=label1)
    # plot_losses(losses2, color='red', label=label2)
    # plot_losses(losses3, color='green', label=label3)

    half_num = len(sys.argv) // 2
    for i in range(0, half_num):
        result = sys.argv[i + 1]
        label = sys.argv[i + half_num + 1]
        print(f"result: {result}, label: {label}")
        losses = parse_result(result)
        color = random_choose_color()
        plot_losses(losses, color=color, label=label)

    save()
