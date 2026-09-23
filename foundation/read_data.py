from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, image_dir, label_dir):
        self.image_dir = os.path.join(root_dir, image_dir)
        self.label_dir = os.path.join(root_dir, label_dir)
        self.img_names = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            label = f.read().strip()
        return img, label

    def __len__(self):
        return len(self.img_names)


root_dir = "C:\\Users\\1\\Desktop\\learn-torch\\dataset\\train"
ants_dataset = MyData(root_dir, "ants_image", "ants_label")
bees_dataset = MyData(root_dir, "bees_image", "bees_label")
img, label = ants_dataset[1]
img.show()
