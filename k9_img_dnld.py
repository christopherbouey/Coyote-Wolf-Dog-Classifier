animals = {'coyote': 'n02114855', 'wolf': 'n02114367', 'dog' : 'n02084071'}

from urllib.request import Request, urlretrieve
import cv2
import numpy as np
import os
import urllib
import sys

def store_raw_images(name, wnid):
    url = f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}'
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    urls = response.read().decode('utf-8')
    attempt = 0
    pic_num = 1
    for i in urls.split('\n'):
        try:
            #print(i)
            urlretrieve(i, f"imgs/{name}/{name}_"+str(pic_num)+".jpg")
            img = cv2.imread(f"imgs/{name}/{name}_"+str(pic_num)+".jpg",1)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (224, 224))
            cv2.imwrite(f"imgs/{name}/{name}_"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
        except Exception as e:
            print(str(e))
        attempt += 1
        print(attempt)
    print(pic_num)

if __name__ == "__main__":
  for i,j in animals.items():
    store_raw_images(i, j)