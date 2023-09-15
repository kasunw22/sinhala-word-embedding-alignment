
#base = "wiki"
base = "cc"

#src = "en"
#tgt = "si"

src = "si"
tgt = "en"

base_dir = "/home/kasunw_22/msc/data"

ref_dict = f"{base_dir}/{base}_{src}-{tgt}_testset_1500_5000_200k.txt"
full_dict_path = f"{base_dir}/{base}-{src}-{tgt}.txt"
out_dict_path = f"{base_dir}/{base}_{src}-{tgt}-full_trainset_wo_test_1500.txt"

with open(ref_dict) as f:
    ref_entries = f.readlines()

with open(full_dict_path) as ff, open(out_dict_path, "w") as fo:
    for i, line in enumerate(ff):
        if line not in ref_entries:
            fo.write(line)

        if (i+1) % 5000 == 0:
            print(f"Processed {i+1} entries..")

print(f"Done writing {out_dict_path}...")
            
