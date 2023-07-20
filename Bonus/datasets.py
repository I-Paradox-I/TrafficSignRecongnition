import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, folder_path, csv_file, transform=None):
        self.folder_path = folder_path
        self.csv_file = csv_file
        self.transform = transform
        self.data = []
        self.classes = []

        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                image_path = os.path.join(folder_path, row['Path'])
                x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])
                class_id = int(row['ClassId'])

                self.data.append((image_path, x1, y1, x2, y2, class_id))
                if class_id not in self.classes:
                    self.classes.append(class_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, x1, y1, x2, y2, class_id = self.data[index]

        # 打开图像
        image = Image.open(image_path)

        # 裁剪图像
        cropped_image = image.crop((x1, y1, x2, y2))

        # 应用数据变换（如果定义了transform）
        if self.transform is not None:
            cropped_image = self.transform(cropped_image)

        # 返回裁剪后的图像和对应的类别
        return cropped_image, class_id
    
    def get_original_image(self, index):
        image_path, x1, y1, x2, y2, class_id = self.data[index]
        image = Image.open(image_path)
        return image
