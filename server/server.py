import serial
import serial.tools.list_ports
import time
import json
import numpy as np
import matplotlib.image as mi
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
from model import AlexNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

port = 'com3'
bps = 115200
flag = False    # 判断串口打开是否成功
data = ''
# 自定义灰度界限，大于这个值为白色，小于这个值为黑色(与原图黑白颠倒)
threshold = 100
# 二值化处理表
table = []
for i in range(256):
    if i < threshold:
        table.append(1)
    else:
        table.append(0)

model = AlexNet()
model.load_state_dict(torch.load('./logs/mnist/best.pth'))
model = model.to(device)

try:
    # 打开串口，并得到串口对象
    ser= serial.Serial(port, bps)
    # 判断是否打开成功
    if (ser.is_open):
        flag = True
        print('Server start!')
except Exception as e:
    print("ERROR: ", e)

if (flag):
    while True:
        time.sleep(0.5)
        data += ser.read_all().decode('ascii')
        if data:
            time.sleep(1)
        if len(data) > 12 and data[0] == '+' and data[-1] == '+':
            data_list = data.split('|')[1]
            data = ''
            data_list = json.loads(data_list)
            data_array = np.array(data_list)
            # print(data_array)
            file_name = f'./image/{int(time.time())}.png'
            mi.imsave(file_name, data_array)

            # 图像预处理
            img = Image.open(file_name).convert('L')
            img = img.crop((1, 1, 98, 98))
            img = img.point(table, '1')
            img.save(file_name)
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))])
            img = transform(img).to(device)
            img = img.unsqueeze(0)

            model.eval()
            with torch.no_grad():
                pred = model(img)
                pred = pred.max(1, keepdim=False)[1]
                result = str(pred.cpu().numpy()[0])
                print(f'Result: {result}')
                ser.write(result.encode('utf-8'))
                print('='*100)
