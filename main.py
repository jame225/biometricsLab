"""
By: James Hassel
Used to run train and test images sets
"""
import preProcessing
import collectData

# PATHS
TRAIN_PATH = "/Users/jameshassel/code/bioMetrics/images/train"
TEST_PATH = "/Users/jameshassel/code/bioMetrics/images/test"
ALL_PATH = "/Users/jameshassel/code/bioMetrics/images/all"


def getImages(flag="TRAIN"):
    """
    Used to return a preprocessed list of images for use in matching
    :param path: Either path to train or test images
    :param flag: Detirmines if train or test is going to be used based on defined global variables, default is set to train
    :return: paired list of images [(f0, s0), (f1, s1), ...]
    """
    if flag.__eq__("TRAIN"):
        trainImages = preProcessing.loadImages(TRAIN_PATH)
        trainPairs = [(trainImages[i], trainImages[i + 1]) for i in range(0, len(trainImages), 2)]
        return trainPairs
    elif flag.__eq__("TEST"):
        testImages = preProcessing.loadImages(TEST_PATH)
        testPairs = [(testImages[i], testImages[i + 1]) for i in range(0, len(testImages), 2)]
        return testPairs
    elif flag.__eq__("ALL"):
        AllImages = preProcessing.loadImages(ALL_PATH)
        AllPairs = [(AllImages[i], AllImages[i + 1]) for i in range(0, len(AllImages), 2)]
        return AllPairs

def main():
    pairs = getImages(flag="ALL")
    collectData.collectData(pairs)


if __name__ == "__main__":
    main()
