# CS2952N-Mini-Project-1
This is the first mini project of CS 2952-N Spring 2023. The project is about multimodal learning.



### Step 1

**Task1:** *Perform zero-shot image classification with CLIP on CIFAR-10. You only need to pick 500 random examples from the validation / test set of CIFAR-10.*

**Task2:** *Perform linear probing with CLIP on CIFAR-10. You need to extract CLIP image embeddings for 1000 random examples from the training set of CIFAR-10, then train a linear classifier. Report classification accuracy on the same 500 random test examples as in Task 1.*



#### Execution

Run the following:

`python Step1.py`



#### Example Result

Upon execution, you'll get a **zero shot accuracy** of **71.60%** and a **linear probing accuracy** of **79.60%** .





### Step 2

**Task 3:** *Convert an image into a list of objects with CLIP. There are several crucial missing pieces for you to figure out: (1) For the zero-shot classification example in Task 1, each image has only a single label. How can you output multiple objects for an image?; (2) What would make a good list of “candidate objects”? You might find it helpful to browse some image examples in the Flickr dataset; (3) Objects only or also attributes? In your list of candidate objects, do you intend to include “cat”, “dog”, or also “cute cat”, “lazy dog”?*

 

**Task 4:** *Convert a list of objects into a sentence with GPT-3. Our notebook already provides some examples for in-context learning, but you would need to decide on the in-context examples to provide to GPT-3.*

 

**Task 5:** *Run your pipeline on at least 10 images from the validation / test set of Flickr 8k, and summarize your observations. Are you happy with what you got? If not, are there things you would like to try to improve the quality of image captions, or do you think there are fundamental limitations on the way we compose CLIP and GPT-3 models?*



#### Prerequisites

Before proceeding to run `Step2.py`, please ensure the following:

1. Download the Flicker8K Datasets:
   - [Image dataset](https://drive.google.com/file/d/1LxsDCy07D6nGkyrmC-74O2fVXKpNzaMG/view?usp=sharing)
   - [Text dataset](https://drive.google.com/file/d/1kaXuTizrLKPurK-2S1zV712664JZqsD8/view?usp=sharing)
2. Directory Setup:
   - Extract both datasets and place them in the `./data` directory.
3. OpenAI API Key Configuration:
   - Ensure you've set your OpenAI API key in `Step2.py` to utilize the API.



#### Execution

Run the following:

`python Step2.py`

Upon execution, captions will be generated for 10 random images from the Flicker 8K test set.



#### How It Works

1. **Iterative Class Identification with CLIP**: 

   The script employs CLIP to identify up to 5 of the most probable existing classes within an image(class names are drawn from the ImageNet dataset). The process begins with an initial prompt, seeking the presence of a specific class. Based on the output, follow-up prompts are deployed to further refine and detect additional classes. This iterative process continues until one of two conditions is met:

   1. An end prompt determines that no more relevant classes are detected in the image.
   2. A maximum of 5 classes have been identified.

2. **GPT-3 Caption Creation**: 

   Post identification, GPT-3 is prompted to craft these identified classes into a meaningful caption. For this caption creation:

   - **In-context Examples**: The script uses examples derived from the Flicker 8K training set. While classes are identified by CLIP for these examples, the corresponding captions come from the dataset's actual ground truth.



#### Example result

| Image Index |                        Top 5 Classes                         |                      Generated Caption                       |                        Actual Caption                        |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      1      |  borzoi, cockroach, African hunting dog, malamute, carousel  |                A borzoi attacks a cockroach.                 |   A group of large white and orange dogs are in the grass.   |
|      2      | Labrador retriever, balance beam, running shoe, hand held computer, picket fence |   A dog wearing a red collar is running on a wooden beam.    |  A blond dog runs down a flight of stairs to the backyard.   |
|      3      | oxcart, beach wagon, Arabian camel, shopping cart, sulphur crested cockatoo |     A man and a woman riding a wagon pulled by a camel.      |   A donkey pulling a cart with a boy in it takes a brake.    |
|      4      | cornet, hair slide, hand held computer, picket fence, chickadee |              A man in a feather hat looks down.              | A lady stands in the middle of a crowd wearing white gloves. |
|      5      | football helmet, thresher, swing, hand held computer, rapeseed |          A boy in a football helmet rides a swing.           | A boy in a yellow uniform carrying a football is blocking another boy in a blue uniform. |
|      6      |      Granny Smith, teddy, bell cote, Newfoundland, bib       |                  A girl holds a teddy bear.                  | A lady holds a little boy while another little boy smiles at them. |
|      7      |    Weimaraner, lakeside, upright, balance beam, cockroach    |           A Weimaraner stands on a lakeside dock.            |        A big hound dog walking on a log in the woods.        |
|      8      |         basketball, shoji, web site, robin, Pembroke         |         A man in a suit shooting a basketball hoop.          |              A boy playing basketball in a gym               |
|      9      |        buckeye, stretcher, quail, spotlight, knee pad        | A man is holding a spotlight while looking at an injured quail. | a football player in a red jersey getting his knee looked at by another man |
|     10      | fire screen, dogsled, table lamp, European fire salamander, motor scooter' |            A boy sitting in front of a fireplace.            |               People sit near a fire outside.                |



#### Analysis of Predictions

The current predictions from the model are below expectations. Some potential causes and corresponding solutions are discussed below:

1. **Challenges with Iterative Class Identification using CLIP**:

   - **Issue**: The class names sourced from ImageNet are notably specific. Despite designing iterative prompts to exclude previously identified classes (e.g., "the photo, besides {}, contains {}"), the CLIP model often predicts several closely related categories for a singular object. This can lead to ambiguity when feeding these classes to GPT-3, especially since Flicker 8K captions lean towards broader category descriptions.

   - **Solution**: Consider sourcing class names directly from the original Flicker 8K dataset. However, given that Flicker 8K's labels are in the form of captions, extracting specific class names while minimizing redundancy will be a meticulous task.

     

2. **Limitations in Composing CLIP and GPT-3**:

   - **Issue**: The integration of CLIP and GPT-3 in the current setup is somewhat rigid, with a one-directional flow of information. Consequently, the data CLIP provides might either be too sparse or too overwhelming for GPT-3, affecting caption quality.
   - **Solution**: Introducing a back-querying mechanism may benefit the captioning model. This iterative feedback could assist GPT-3 in acquiring the most relevant details necessary for accurate captioning.



