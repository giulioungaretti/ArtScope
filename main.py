from brain import *
target = read_one_single_feature("prod_technique", "KMS3924")
df = read_all_set_features(["prod_technique",  "objectnumber", "title_dk"], 6000)
df  = d(target, df,"prod_technique")
print df.head()