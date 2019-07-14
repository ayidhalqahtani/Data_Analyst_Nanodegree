
# coding: utf-8

# In[2]:

import xml.etree.cElementTree as ET
import pprint
from collections import defaultdict


def count_tags(filename):
    t = {}
    for _,elem in ET.iterparse(filename):
        if elem.tag in t:
            t[elem.tag] +=1
        else:
            t[elem.tag] =1
    return t




def test():

    tags = count_tags('Baltimore_sample.osm')
    pprint.pprint(tags)
    
    

if __name__ == "__main__":
    test()


# In[ ]:



