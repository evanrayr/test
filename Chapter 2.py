#!/usr/bin/env python
# coding: utf-8

# In[1]:


#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()


# In[2]:


#hide
from fastbook import *
from fastai.vision.widgets import *


# In[3]:


key = os.environ.get('AZURE_SEARCH_KEY', 'fecb24e7bb234d45aab31258e64a6753')


# In[4]:


search_images_bing


# In[5]:


results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('content_url')
len(ims)


# In[7]:


doc(os.environ.get)


# In[6]:


bear_types = 'grizzly','black','teddy'
path = Path('bears') #must be defined as a path item


# In[7]:


if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))


# In[8]:


fns = get_image_files(path)
fns


# In[9]:


failed = verify_images(fns)
failed
len(failed)


# In[10]:


failed.map(Path.unlink);


# In[11]:


class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])


# In[12]:


bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[13]:


dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[14]:


bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[15]:


bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[16]:


bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)


# In[17]:


bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)


# In[18]:


bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)


# In[19]:


learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


# In[20]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[21]:


interp.plot_top_losses(5, nrows=1)


# In[22]:


#hide_output
cleaner = ImageClassifierCleaner(learn)
cleaner


# In[ ]:


learn.export()


# In[6]:


path = Path() #initialize empty path object
path.ls(file_exts='.pkl') #the ls method verifies this path


# In[7]:


learn_inf = load_learner(path/'export.pkl')


# In[8]:


ims = ['http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg']
dest = 'grizzly.jpg'
download_url(ims[0], dest)
im = Image.open(dest)
im.to_thumb(128,128)

learn_inf.predict('grizzly.jpg')


# In[41]:


# list the categories of the dependent variable
learn_inf.dls.vocab


# In[9]:


#hide_output
btn_upload = widgets.FileUpload()
btn_upload


# In[11]:


img = PILImage.create(btn_upload.data[-1])


# In[12]:


#hide_output
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[14]:


learn_inf.predict(img)
pred,pred_idx,probs = learn_inf.predict(img) #need this syntax to define an array output into multiple objects


# In[15]:


#hide_output
lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}' 
    # f' seems to preface text
    # probs is like a function which we can assign the predicted class index to
    #.04 is decimal places behind coma 
lbl_pred


# In[16]:


probs[0]


# In[17]:


btn_run = widgets.Button(description='Classify')
btn_run


# In[18]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[19]:


#hide
#Putting back btn_upload to a widget for next cell
btn_upload = widgets.FileUpload()


# In[20]:


#hide_output
VBox([widgets.Label('Select your bear!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[21]:


get_ipython().system('pip install voila')
get_ipython().system('jupyter serverextension enable --sys-prefix voila')


# In[ ]:





# In[ ]:


##########################################
##################TRIALS##################
##########################################


# In[22]:


#download images for 3 types of bags
bag_types = 'hand','tote','sling'
path = Path('bags')
if not path.exists():
    path.mkdir()
    for o in bag_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bag')
        download_images(dest, urls=results.attrgot('contentUrl'))
        
#store in object
fns = get_image_files(path)
failed = verify_images(fns)
failed
len(failed)

#unlink unverified
failed.map(Path.unlink);

#initialize datablock
bags = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

#set bags as dataloaders item
dls = bags.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[4]:


#download images for 3 types of bags
bp_types = 'lisa','jennie','rosé','jisoo'
path = Path('bp')
if not path.exists():
    path.mkdir()
    for o in bp_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bp')
        download_images(dest, urls=results.attrgot('contentUrl'))
        
#store in object
fns = get_image_files(path)
failed = verify_images(fns)
failed
len(failed)

#unlink unverified
failed.map(Path.unlink);

#initialize datablock
bp = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

#augment images
bp = bp.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())

#set bags as dataloaders item
dls = bp.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[5]:


#train model
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


# In[8]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[9]:


uploader = widgets.FileUpload()
uploader


# In[ ]:


#predict
learn.export()
path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')

img = PILImage.create(uploader.data[0])

learn_inf.predict(img)


# In[54]:


img = PILImage.create(uploader.data[0])
learn_inf.predict(img)


# In[60]:




