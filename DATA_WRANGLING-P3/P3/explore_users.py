
# coding: utf-8

# In[2]:

#!/usr/bin/env python

import xml.etree.cElementTree as ET
import pprint
import re
 

def get_user(element):
    if ('uid') in element.attrib:

      return element.attrib['uid']


def process_map(filename):
    user = set()
    for _, element in ET.iterparse(filename):
        uid=get_user(element)
        if uid != None:
           user.add(uid)

    return user


def test():

    users = process_map('Baltimore_sample.osm')
    pprint.pprint(users)
    



if __name__ == "__main__":
    test()


# In[ ]:



