from brain import *
from sklearn.cluster import KMeans


target = read_one_single_feature("prod_technique", "KMS3924")
df = read_all_set_features(["prod_technique",   "objectnumber", "title_dk",
                            "object_type", "externalurl", "artists_natio", "object_production_date"])


# add special features
feats = ["prod_technique", "object_type"]

for feat in feats:
    df = d(read_one_single_feature(feat, "KMS3924"), df, feat)


target = read_one_single_feature("artists_natio", "KMS3924")
add_location(target, df)

# add year

add_production_date(df)
target = read_one_single_feature("object_production_date", "KMS3924")
target = int(string_date(target)[0])
score_dates(target, df)
# fix problem
df['score-object_production_date'][df.object_production_date == 1570] = 1
df = df.dropna()

X = df[[u"score-prod_technique", u"score-object_type",
        u"score-artists_natio", u"score-object_production_date"]]

cluster = 5
k_means = KMeans(n_clusters=cluster, random_state=10)

k_means.fit(X)
y_pred = k_means.predict(X)
df["group"] = y_pred

crit = df[[u"score-prod_technique", u"score-object_type",
           u"score-artists_natio", u"score-object_production_date"]].mean(axis=1)
df['sort_axis'] = crit
df = df.set_index(df.sort_axis)
df = df.sort()

f1 = open('./res.csv', 'w+')
for i in df.group.unique():
    temp = df[df.group == i]
    name = temp[-1:].objectnumber.values
    title = temp[-1:].title_dk.values
    url = temp[-1:].externalurl.values
    score = temp[-1:].sort_axis.values
    # print score
    downloader(name[0].split('/')[0], str(url[0]))
    string = name[0].split('/')[0] + '.jpg' + ',' + title
    print>> f1, string + '\n'
f1.close()

f1 = open('./res.csv', 'a')
for i in df.group.unique():
    temp = df[df.group == i]
    name = temp[:1].objectnumber.values
    title = temp[:1].title_dk.values
    url = temp[:1].externalurl.values
    score = temp[:1].sort_axis.values
    # print score
    downloader(name[0].split('/')[0], str(url[0]))
    string = name[0].split('/')[0] + '.jpg' + ',' + title
    print>> f1, string + '\n'

f1.close()


# best match
df[df.group = 4]
