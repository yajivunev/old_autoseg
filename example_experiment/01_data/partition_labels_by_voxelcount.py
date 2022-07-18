import os
import sys
import json
import random

"""
    Script to return list of unique IDs such that they are 
    a specific fraction of all the labels within labels_mask.

    Basically, partitioning the set of objects into required 
    number of subsets where each subset's total size is a 
    specified fraction of the set's size.
"""


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    sparsity = float(sys.argv[2])

    ids_counts = os.path.join(input_zarr,'ids_counts.json')

    with open(ids_counts,"r") as f:
        by_counts = json.load(f) #keys are str(unique_id)

    by_object = {v['name']:{'unique':v['unique'],'count':int(k)} for k,v in by_counts.items()}
    by_unique = {v['unique']:{'name':v['name'],'count':int(k)} for k,v in by_counts.items()}

    #sort keys(counts), small to big
    counts = sorted(list(map(int,by_counts.keys())))
    total_voxels = sum(counts)

    available = int(sparsity*total_voxels)

    all_uniques = [by_counts[str(x)]['unique'] for x in counts if x > 200000]

    partitions = dict()

    print(f"AVAILABLE: {available}")
    for i in range(5):

        print(i)
        current_sum = 0

        partition = []

        while current_sum <= available:
           
            if current_sum/available > 0.95:
                break
            
            diff = available - current_sum

            #pick a random object
            #unique = random.choice(all_uniques)
            unique = all_uniques.pop(-1)

            count = by_unique[unique]['count']
            obj = by_unique[unique]['name']

            #update
            if count < diff:

                current_sum += count
                partition.append({'name':obj,'unique':unique,'count':count})
                print(current_sum)

            else:

                all_uniques.insert(0,unique)

        partitions[i] = partition

    print("final partitions:")
    print(partitions)

    with open(os.path.join(input_zarr,f"sparse_{sparsity}.json"),"w") as f:
        json.dump(partitions,f,indent=4)
