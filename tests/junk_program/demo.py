import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# サンプルデータの作成
np.random.seed(42)
data = pd.DataFrame(
    {
        "Category": np.repeat(["A", "B", "C", "D"], 100),
        "Value": np.concatenate(
            [
                np.random.normal(0, 1, 100),
                np.random.normal(1, 1.5, 100),
                np.random.normal(-1, 0.5, 100),
                np.random.normal(2, 2, 100),
            ]
        ),
    }
)

# バイオリンプロットの作成
sns.violinplot(x="Category", y="Value", data=data)

# X軸の注釈を追加
categories = data["Category"].unique()
annotations_1 = ["注釈1_A", "注釈1_B", "注釈1_C", "注釈1_D"]
annotations_2 = ["注釈2_A", "注釈2_B", "注釈2_C", "注釈2_D"]

for idx, category in enumerate(categories):
    plt.text(idx, plt.ylim()[0] - 0.5, annotations_1[idx], ha="center")
    plt.text(idx, plt.ylim()[0] - 1.0, annotations_2[idx], ha="center")

# y軸のリミットを変更して注釈が見えるようにする
plt.ylim(plt.ylim()[0] - 1.5, plt.ylim()[1])

plt.show()
