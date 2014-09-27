from urllib2 import urlopen
from jellyfish import jaro_distance
import pandas as pd
import numpy as np


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
    features = rsp["response"]["docs"]
    return features


def read_one_single_feature(feature, id):
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
    if type(artworks[feature]) == list:
        return artworks[feature][0]
    else:
        return  artworks[feature]


def read_all_single_feature(feature, lim=64561):
    '''
    feature: string i.e "prod_technique"
    lim:  int number of records to parse default all of them
    ----
    returns a dictionary containing the string of the feature of the artwork
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

def get_related_artworks():
    """
    not implemented yet
    """
    target_url =   "http://solr.smk.dk:8080/solr-h4dk/prod_search_pict/select?q=q:*KMS3924*&wt=json"

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
            response = []
            for artwork in artworks:
                try:
                    artwork[feature]
                    if type(artwork[feature]) == list:
                        if len(artwork[feature][0]) > 2:
                            response.append(artwork[feature][0])
                        else:
                            response.append(np.nan)
                    else:
                        if len(artwork[feature]) > 2:
                            response.append(artwork[feature])
                        else:
                            response.append(np.nan)
                except:
                    response.append(np.nan)
            data[str(feature)] = response
        return pd.DataFrame.from_dict(data).dropna()
    else:
        print("not a list")


def d(target, DataFrame, feature):
    '''
    returns a score of similarity where 1 is the same and 0 is totaly
    different
    '''
    DataFrame['score-{0}'.format(feature)] = DataFrame[feature].apply(
        lambda row:  jaro_distance(target, row))
    return DataFrame

def add_location(target,  DataFrame, feature):
        feature = artists_natio
        DataFrame['score-{0}'.format(feature)] = DataFrame[feature].apply(
        lambda row:  jaro_distance(target, row))
