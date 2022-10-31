#!/usr/bin/env python
# coding: utf-8

# In[50]:


# !kaggle --help
import os
print(os.getcwd())
base_path = 'C:\\Users\\DELL\\Documents\\AI'
os.chdir(base_path)
print(os.getcwd())


# In[4]:


import socket, warnings
try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))

except socket.error as ex: raise Exception("STOP: No internet. Click '>|' in the top right and set 'Internet' switch to on")


# In[26]:


import os
# iskaggle = os.environ.get('HOMEPATH')

# if iskaggle:
#     get_ipython().system('pip install -Uqq fastai')


# In[27]:


import fastai


# In[33]:


# get_ipython().system('pip install -Uqq duckduckgo_search')


# In[34]:


# get_ipython().system('pip install pathlib ruamel-yaml')


# In[35]:


from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=200): return L(ddg_images(term, max_results=max_images)).itemgot('image')


# In[67]:


urls = search_images('bird photos', max_images=1)
urls[0]


# In[37]:


from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=True)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256, 256)


# In[73]:


download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)


# In[68]:


def get_image(string):
    download_url(search_images(f'{string} photos', max_images=1)[0],
                 f'{string}.jpg', show_progress=False)
    Image.open(f'{string}.jpg').to_thumb(256,256)
    return f'{string}.jpg'

def get_image_url(string):
    urls = search_images(f'{string} photos', max_images=1)
    return urls[0]
    


# In[77]:


url = get_image_url('rose')
url


# In[78]:


def get_sample_image(url):
    img = requests.get(url)
    file = open("sample_image.jpg", "wb")
    file.write(img.content)
    file.close()
    img_file_name = "sample_image.jpg"
    return img_file_name

get_sample_image(url)


# In[39]:


searches = 'forest','bird'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)


# In[40]:


Path('bird_or_not/bird').absolute()


# In[51]:


failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# In[52]:


get_image_files(path)


# In[54]:


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)


# In[55]:


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


# In[62]:


is_bird, _, probs = learn.predict(PILImage.create('rose.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")


# In[59]:


import pickle
pickle.dump(learn, open('bird_clf.pkl', 'wb'))


# In[70]:


import streamlit as st
from PIL import Image
import requests


# In[79]:


st.title("Image Classification")
st.write("Using ResNet Model to classify the image")

url = st.text_input("Enter Image URL:")

if url:
    image = get_sample_image(url)
    st.image(image)
    classify = st.button("classify image")
    if classify:
        st.write("")
        st.write("Classifying")
        label = learn(image)
        st.write(f"{label[1]}  {label[2]:0.2f}")
else:
    st.write("Paste Image URL")


# In[ ]:




