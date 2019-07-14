
# coding: utf-8

# In[3]:

#!/usr/bin/env python

import xml.etree.cElementTree as ET
import pprint
import re



lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')


def key_type(element, keys):
    if element.tag == "tag":
       if re.search(problemchars, element.attrib['k']):
            keys['problemchars'] += 1
       elif re.search(lower, element.attrib['k']):
            keys['lower'] += 1
       elif re.search(lower_colon, element.attrib['k']):
            keys['lower_colon'] += 1
       
       else:
            keys['other'] += 1
    return keys


def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys



def test(): 
    keys = process_map('Baltimore_sample.osm')
    pprint.pprint(keys)
 


if __name__ == "__main__":
    test()


# In[ ]:



