from brain import *
from sklearn.cluster import KMeans


target = read_one_single_feature("prod_technique", "KMS3924")
df = read_all_set_features(["prod_technique",   "objectnumber", "title_dk", "object_type", "externalurl", "artists_natio", "object_production_date"])


# add special features
feats = ["prod_technique", "object_type"]

for feat in feats:
    df  = d(read_one_single_feature(feat, "KMS3924"), df,feat)


target = read_one_single_feature("artists_natio", "KMS3924")
add_location(target, df)

# add year

add_production_date(df)
target = read_one_single_feature("object_production_date", "KMS3924")
target = int(string_date(target)[0])
score_dates(target, df)

df = df.dropna()


X = df[[u"score-prod_technique", u"score-object_type", u"score-artists_natio", u"score-object_production_date"]]

cluster = 5
k_means = KMeans(n_clusters=cluster,random_state=10)

k_means.fit(X);
y_pred = k_means.predict(X)
df["group"] = y_pred

