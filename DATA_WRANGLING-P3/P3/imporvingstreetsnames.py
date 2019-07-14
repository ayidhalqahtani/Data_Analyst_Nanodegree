
# coding: utf-8

# In[4]:

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "Baltimore_sample1.osm"
#
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]


mapping = { "St": "Street",
            "St.": "Street",
            "street": "Street",
            "Ave":"Avenue",
            "Blvd.":"Boulevard",
            "Blvd":"Boulevard",
            "Rd":"Road",
            'Hwy': "Highway",
             'Hwy.': "Highway",
             "highway":"Highway",
             'CIrcle':'Circle',
             'Ct.':'Court',
             'Ct':"Court",
             'Hwy':"Highway",
             'Westparkway':"West Parkway"
           
           
            }


def audit_street_type(street_types, street_name):
    k = street_type_re.search(street_name)
    if k:
        Stype = k.group()
        if Stype not in expected:
            street_types[Stype].add(street_name)


def is_street_name(elem):
     return (elem.tag == "tag") and (elem.attrib['k'] == "addr:street")
 

def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types

updated_streets = {}

def update_name(name, mapping):
   k = street_type_re.search(name)
   if k:
        type_of_street = k.group()
        if type_of_street in mapping:
         new = street_type_re.sub(mapping[type_of_street],name)
         updated_streets[name] = new
         return new



    

def test():
    st_types = audit(OSMFILE)
    
    pprint.pprint(dict(st_types))

    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            print name, "==>", better_name
         


if __name__ == '__main__':
    test()


# In[ ]:



