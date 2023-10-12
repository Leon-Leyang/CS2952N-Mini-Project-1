# CS2952N-Mini-Project-1
This is the first mini project of CS 2952-N Spring 2023. The project is about multimodal learning.



### Step 1

**Task1:** *Perform zero-shot image classification with CLIP on CIFAR-10. You only need to pick 500 random examples from the validation / test set of CIFAR-10.*

**Task2:** *Perform linear probing with CLIP on CIFAR-10. You need to extract CLIP image embeddings for 1000 random examples from the training set of CIFAR-10, then train a linear classifier. Report classification accuracy on the same 500 random test examples as in Task 1.*



Run 'Step1.py', you'll get a **zero shot accuracy** of **71.60%** and a **linear probing accuracy** of **79.60%** .





### Step 2

**Task 3:** *Convert an image into a list of objects with CLIP. There are several crucial missing pieces for you to figure out: (1) For the zero-shot classification example in Task 1, each image has only a single label. How can you output multiple objects for an image?; (2) What would make a good list of “candidate objects”? You might find it helpful to browse some image examples in the Flickr dataset; (3) Objects only or also attributes? In your list of candidate objects, do you intend to include “cat”, “dog”, or also “cute cat”, “lazy dog”?*

 

**Task 4:** *Convert a list of objects into a sentence with GPT-3. Our notebook already provides some examples for in-context learning, but you would need to decide on the in-context examples to provide to GPT-3.*

 

**Task 5:** *Run your pipeline on at least 10 images from the validation / test set of Flickr 8k, and summarize your observations. Are you happy with what you got? If not, are there things you would like to try to improve the quality of image captions, or do you think there are fundamental limitations on the way we compose CLIP and GPT-3 models?*



