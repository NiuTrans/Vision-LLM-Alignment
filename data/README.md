The source of the dataset is described below:

- Image Folder: Download 2017 Train images from ```https://cocodataset.org/#download```.

- SFT Dataset: Download from [LLaVA-Instruct](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K). Like LLaVA, we can also pre-train for feature alignment using SFT datasets, such as CC3M.

- RM Dataset: Download from [LLaVA-Human-Preference](https://huggingface.co/datasets/zhiqings/LLaVA-Human-Preference-10K). We use the ```convert_to_lava.py``` script to convert it into the format of our reward training dataset.

- PPO Dataset: We use the same data format as SFT, but in this answer field we can leave it empty or give it a random string.


