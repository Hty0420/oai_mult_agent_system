from src.algorithm.data_augmentation import OfflineAugmentor
oa = OfflineAugmentor()
oa.augment_dataset(csv_path=r"./datasets/labels_train.csv",
                   output_dir=r"augment_datasets_train")