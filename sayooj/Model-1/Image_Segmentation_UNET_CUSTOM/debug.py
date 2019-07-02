from data_loader_rle import SIIMDataset
import PIL
import matplotlib.pyplot as plt
import numpy as np
csv_path = '/home/sayooj/Downloads/cloned-repos/Image_Segmentation-master/siimdata/train-rle.csv'
data_path = '/home/sayooj/Downloads/cloned-repos/Image_Segmentation-master/siimdata/train'
dataset = SIIMDataset(img_dir=data_path, df_path=csv_path)

print(len(dataset))
image , encode = dataset[5]
encode = np.squeeze(encode)
print(image.size)
print(encode.shape)
print(type(image))
print(type(encode))
# plt.imshow(image)
# plt.imshow(encode)

plt.show()