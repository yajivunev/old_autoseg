import daisy
import numpy as np
import sys

if __name__ == "__main__":

    input_file = sys.argv[1]
    datasets = sys.argv[2:]

    print(f"\n{input_file}: ")

    for dataset in datasets:

        try:
            ds = daisy.open_ds(input_file,dataset)
        except:
            print(f"{dataset} not found in {input_file}, going to next..")
            pass

        arr = ds.to_ndarray()

        uniques = np.unique(arr,return_counts=True)

        print(f"{dataset}: uniques = {uniques}, shape = {arr.shape}, ratio = {uniques[1][1]/(uniques[1][0]+uniques[1][1])}")
