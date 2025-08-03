from dataset.constants import (
    HEAD_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
)
import pandas as pd


def eval_cats(csv_path):
    def read_csv(file_path):
        df = pd.read_csv(file_path)
        # 读取头部类别
        head_data = df[df["class"].isin(HEAD_CATS_SCANNET_200)]
        common_data = df[df["class"].isin(COMMON_CATS_SCANNET_200)]
        tail_data = df[df["class"].isin(TAIL_CATS_SCANNET_200)]
        return head_data, common_data, tail_data

    head_data, common_data, tail_data = read_csv(csv_path)
    print(head_data)
    print("=" * 100)
    print(common_data)
    print("=" * 100)
    print(tail_data)

    print(head_data[["ap", "ap50", "ap25"]].mean())
    print(common_data[["ap", "ap50", "ap25"]].mean())
    print(tail_data[["ap", "ap50", "ap25"]].mean())
