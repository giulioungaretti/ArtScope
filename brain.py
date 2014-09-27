from urllib2 import urlopen
from jellyfish import jaro_distance
import pandas as pd


def read_single_artwork(id):
    '''

    takes an string id which corresponds to the id in the smk museum
    target is KMS3924
    ----
    returns a dictionary containing the fields that describe the artwork
    '''
    target_url = "http://solr.smk.dk:8080/solr-h4dk/prod_collection/select?q=id%3A{0}&wt=json".format(
        id)
    conn = urlopen(target_url)
    rsp = eval(conn.read())
    features = rsp["response"]["docs"][0]
    return features


def read_one_single_feature(feature, id, lim=64561):
    '''
    feature: string i.e "prod_technique"
    lim:  int number of records to parse default all of them
    ----
    returns a dictionary containing the fields that describe the artwor
    '''
    target_url = "http://solr.smk.dk:8080/solr-h4dk/prod_collection/select?q=id%3A{0}&wt=json".format(
        id)
    conn = urlopen(target_url)
    rsp = eval(conn.read())
    artworks = rsp["response"]["docs"][0]
    try:
        artworks[feature]
        artworks[feature][0]
    except:
        print 'feature not there'
    return artworks[feature][0]


def read_all_single_feature(feature, lim=64561):
    '''
    feature: string i.e "prod_technique"
    lim:  int number of records to parse default all of them
    ----
    returns a dictionary containing the fields that describe the artwor
    '''
    target_url = "http://solr.smk.dk:8080/solr-h4dk/prod_collection/select?q=*%3A*&rows={0}&wt=json".format(
        lim)
    conn = urlopen(target_url)
    rsp = eval(conn.read())
    artworks = rsp["response"]["docs"]
    response = []
    for artwork in artworks:
        try:
            artwork[feature]
            response.append(artwork[feature][0])
        except:
            continue
    return response


def read_all_set_features(features, lim=64561):
    '''
    feature: list of strings i.e "prod_technique" "objectnumber",
    lim:  int number of records to parse default all of them
    ----
    returns a dictionary containing the fields that describe the artwork
    ---
    example:
    read_all_set_features(["prod_technique",  "objectnumber", "title_dk"])
    '''
    if type(features) == list:
        data = {}
        target_url = "http://solr.smk.dk:8080/solr-h4dk/prod_collection/select?q=*%3A*&rows={0}&wt=json".format(
        lim)
        conn = urlopen(target_url)
        rsp = eval(conn.read())
        artworks = rsp["response"]["docs"]
        for feature in features:
            print feature
            response = []
            for artwork in artworks:
                try:
                    artwork[feature]
                    if  type(artwork[feature]) == list:
                        response.append(artwork[feature][0])
                    else:
                        response.append(artwork[feature])
                except:
                    response.append(None)
            data[str(feature)] =  response
        return pd.DataFrame.from_dict(data)
    else:
        print "not a list"

def d(target, DataFrame, feature):
    DataFrame['score'] = DataFrame[feature].apply(lambda row: 1 - jaro_distance(target, row))
    return DataFrame
