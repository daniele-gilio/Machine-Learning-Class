from pvml import svm
import image_features
import glob
from matplotlib import image

def file_list(path): #pn stands for positive/negative
    files = [f for f in glob.glob(path + "/" + "**/*.jpg", recursive=True)]
    return files

classes=["bluebell", "buttercup", "colts-foot", "daisy", "dandelion", "fritillary",
            "iris", "lili-valley", "pansy", "sunflower", "tigerlily", "windflower"]
path="flowers"
n_c=len(classes)

file=file_list(path)
train_size=60*12
test_size=20*12
print(image_features.rgb_cooccurrence_matrix(image.imread(file[5])).shape)
