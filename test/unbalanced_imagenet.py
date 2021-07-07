from src.multicropdataset import MultiCropDataset, build_label_index
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/datasets_local/ImageNet/train",
                    help="path to dataset repository")
parser.add_argument("--subset", type=int, default=-1,
                    help="take a fix number of images per class (example 260)")
parser.add_argument("--ratio_minority_class", type=float, default=0.)
parser.add_argument("--ratio_step_size", type=float, default=1.)
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

args = parser.parse_args()

train_dataset = MultiCropDataset(
    args.data_path,
    args.size_crops,
    args.nmb_crops,
    args.min_scale_crops,
    args.max_scale_crops,
    return_index=True,
    subset=args.subset,
    ratio_minority_class=args.ratio_minority_class,
    ratio_step_size=args.ratio_step_size,
)
plt.hist(train_dataset.targets, bins=len(train_dataset.classes))
plt.show()
print("End")
