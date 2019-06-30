from data_loader_rle import SIIMDataset

csv_path = '/home/sayooj/Downloads/cloned-repos/Image_Segmentation-master/siimdata/train-rle.csv'
data_path = '/home/sayooj/Downloads/cloned-repos/Image_Segmentation-master/siimdata/train'
dataset = SIIMDataset(csv_path, data_path)

print(len(dataset))
image , encode = dataset[5]