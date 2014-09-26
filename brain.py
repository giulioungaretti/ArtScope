from urllib2 import urlopen


def read_single_artwork(id):
    '''

    takes an string id which corresponds to the id in the smk museum
    ----
    returns a dictionary containing the fields that describe the artwork
    '''
    target_url = "http://solr.smk.dk:8080/solr-h4dk/prod_collection/select?q=id%3A{0}&wt=json&indent=true".format(id)
    conn = urlopen(target_url)
    rsp = eval(conn.read())
    features = rsp["response"]["docs"][0]
    return features

