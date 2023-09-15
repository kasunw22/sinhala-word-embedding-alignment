dataset_names = ["cc_en-si_trainset_1500_5000_200k.txt", "cc_en-si_testset_1500_5000_200k.txt", "wiki_en-si_trainset_1500_5000_200k.txt", "wiki_en-si_testset_1500_5000_200k.txt"]
output_names = ["cc_si-en_trainset_1500_5000_200k.txt", "cc_si-en_testset_1500_5000_200k.txt", "wiki_si-en_trainset_1500_5000_200k.txt", "wiki_si-en_testset_1500_5000_200k.txt"]
max_unique_sizes = [5000, 1500, 5000, 1500]
max_same = 5

base_dir = "../data"

swapped_entries = {}

for di, do, N in zip(dataset_names, output_names, max_unique_sizes):
    print(f"Processing {di}...")
    dataset_path = f"{base_dir}/{di}"
    output_name = f"{base_dir}/{do}"
    
    fi = open(dataset_path)
    fo = open(output_name, "w")
    
    swapped_entries = {}

    for line in fi:
        src, tgt = line.strip().split()
        #fo.write(f"{tgt} {src}\n")

        if tgt not in swapped_entries:
            swapped_entries[tgt] = []

        swapped_entries[tgt].append(src)
    
    fi.close()

    print("Writing the result...")
    
    fo = open(output_name, "w")
    
    for i, (src, tgt_words) in enumerate(swapped_entries.items()):    
        for tgt in tgt_words[:max_same]:
            fo.write(f"{src} {tgt}\n")

        if i >= N:
            break
    
    fo.close()
    
    print("Writing done...\n")

print("Done!")
