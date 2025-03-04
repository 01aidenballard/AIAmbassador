import json

# load our dataset as ds, 'r' means we are reading from the file provided
with open('code/dataset.json', 'r') as dataset:
    ds = json.load(dataset)

# tally variable
total_qas = 0

# print statements
print("QA pairs: ")
for x in range(0,len(ds["data"])):
    print(f"    {ds["data"][x]["title"]}: {len(ds["data"][x]["paragraphs"][0]["qas"])}")
    total_qas += len(ds["data"][x]["paragraphs"][0]["qas"])


print(f"Total: {total_qas}")