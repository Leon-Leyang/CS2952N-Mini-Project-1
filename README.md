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



#### Example Result

The table provides a summary of details for each of the 10 examples, including the image path, the top five identified classes, the generated caption, and the corresponding actual caption.

|        Image  Path        |                        Top 5 Classes                         |                      Generated Caption                       |                        Actual Caption                        |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2216695423_1362cb25f3.jpg | borzoi, hand held computer, African hunting dog, Saluki, wire haired fox terrier |                A dog runs through the grass.                 |   A group of large white and orange dogs are in the grass.   |
| 246055693_ccb69ac5c6.jpg  | Chesapeake Bay retriever, running shoe, balance beam, computer keyboard, picket fence |     A dog lies on a grassy lawn, looking at the camera.      |  A blond dog runs down a flight of stairs to the backyard.   |
| 2265096094_8cc34d669c.jpg |    horse cart, oxcart, Maltese dog, beach wagon, Model T     |              A man driving a horse-drawn cart.               |   A donkey pulling a cart with a boy in it takes a brake.    |
| 2204550058_2707d92338.jpg | groenendael, hand held computer, picket fence, neck brace, ringlet |               A dog with a ball in his mouth.                | A lady stands in the middle of a crowd wearing white gloves. |
| 2526041608_a9775ab8d7.jpg | football helmet, desktop computer, swing, scoreboard, shield |                   A boy playing football.                    | A boy in a yellow uniform carrying a football is blocking another boy in a blue uniform. |
| 2522297487_57edf117f7.jpg | cradle, cougar, Dandie Dinmont, spaghetti squash, Granny Smith |                 A baby sleeping in a cradle.                 | A lady holds a little boy while another little boy smiles at them. |
| 1554713437_61b64527dd.jpg | Weimaraner, balance beam, cocktail shaker, worm fence, Greater Swiss Mountain dog |             A dog is walking on a balance beam.              |        A big hound dog walking on a log in the woods.        |
| 2251747182_6b67a3ab8b.jpg |      basketball, web site, shoji, Pembroke, ballplayer       |             A girl in a kimono plays basketball.             |              A boy playing basketball in a gym               |
| 241345905_5826a72da1.jpg  |      buckeye, stretcher, prayer rug, scoreboard, Saluki      |               A man stands on a diving board.                | a football player in a red jersey getting his knee looked at by another man |
| 2610447973_89227ff978.jpg | black grouse, table lamp, mountain tent, space heater, tripod | A man and his dog sit at a table in a black grouse hunting cabin. |               People sit near a fire outside.                |



Below are the visualizations:

![2216695423_1362cb25f3](./result/2216695423_1362cb25f3.jpg)  
**Top 5 Classes:** borzoi, hand held computer, African hunting dog, Saluki, wire haired fox terrier  
**Generated Caption:** A dog runs through the grass.  
**Actual Caption:** A group of large white and orange dogs are in the grass.

---

![246055693_ccb69ac5c6](./result/246055693_ccb69ac5c6.jpg)  
**Top 5 Classes:** Chesapeake Bay retriever, running shoe, balance beam, computer keyboard, picket fence  
**Generated Caption:** A dog lies on a grassy lawn, looking at the camera.  
**Actual Caption:** A blond dog runs down a flight of stairs to the backyard.

---

![2265096094_8cc34d669c](./result/2265096094_8cc34d669c.jpg)  
**Top 5 Classes:** horse cart, oxcart, Maltese dog, beach wagon, Model T  
**Generated Caption:** A man driving a horse-drawn cart.  
**Actual Caption:** A donkey pulling a cart with a boy in it takes a brake.

---

![2204550058_2707d92338](./result/2204550058_2707d92338.jpg)  
**Top 5 Classes:** groenendael, hand held computer, picket fence, neck brace, ringlet  
**Generated Caption:** A dog with a ball in his mouth.  
**Actual Caption:** A lady stands in the middle of a crowd wearing white gloves.

---

![2526041608_a9775ab8d7](./result/2526041608_a9775ab8d7.jpg)  
**Top 5 Classes:** football helmet, desktop computer, swing, scoreboard, shield  
**Generated Caption:** A boy playing football.  
**Actual Caption:** A boy in a yellow uniform carrying a football is blocking another boy in a blue uniform.

---

![2522297487_57edf117f7](./result/2522297487_57edf117f7.jpg)  
**Top 5 Classes:** cradle, cougar, Dandie Dinmont, spaghetti squash, Granny Smith  
**Generated Caption:** A baby sleeping in a cradle.  
**Actual Caption:** A lady holds a little boy while another little boy smiles at them.

---

![1554713437_61b64527dd](./result/1554713437_61b64527dd.jpg)  
**Top 5 Classes:** Weimaraner, balance beam, cocktail shaker, worm fence, Greater Swiss Mountain dog  
**Generated Caption:** A dog is walking on a balance beam.  
**Actual Caption:** A big hound dog walking on a log in the woods.

---

![2251747182_6b67a3ab8b](./result/2251747182_6b67a3ab8b.jpg)  
**Top 5 Classes:** basketball, web site, shoji, Pembroke, ballplayer  
**Generated Caption:** A girl in a kimono plays basketball.  
**Actual Caption:** A boy playing basketball in a gym.

---

![241345905_5826a72da1](./result/241345905_5826a72da1.jpg)  
**Top 5 Classes:** buckeye, stretcher, prayer rug, scoreboard, Saluki  
**Generated Caption:** A man stands on a diving board.  
**Actual Caption:** a football player in a red jersey getting his knee looked at by another man.

---

![2610447973_89227ff978](./result/2610447973_89227ff978.jpg)  
**Top 5 Classes:** black grouse, table lamp, mountain tent, space heater, tripod  
**Generated Caption:** A man and his dog sit at a table in a black grouse hunting cabin.  
**Actual Caption:** People sit near a fire outside.



#### Analysis of Predictions

The model's current predictions sometimes yield overly simplified captions that lack detail. For instance, it captions an image as "A man driving a horse-drawn cart," whereas the actual description is "A donkey pulling a cart with a boy in it takes a brake." Furthermore, while it may describe a scene as "A dog runs through the grass," the actual scene involves "A group of large white and orange dogs in the grass." There are also instances where the results are far off the mark, such as predicting "A dog with a ball in his mouth" for an image whose accurate description is "A lady stands in the middle of a crowd wearing white gloves."



Some potential causes and corresponding solutions are discussed below:

1. **Zero-shot Identification with CLIP**

   - **Issue**: The class names used are derived from ImageNet, which might not align seamlessly with the nuances of Flicker 8K.

   - **Solution**: An approach worth considering is to source class names directly from the Flicker 8K dataset. However, since Flicker 8K's labels are caption-based, extracting distinct class names and reducing overlaps will require careful processing.

     

2. **Deficiency of Adjectives and Verbs in CLIP Output**:

   - **Issue**: ImageNet is primarily centered around nouns, missing out on adjectives and verbs. This leads to situations where, for a scene like "A boy in a yellow uniform carrying a football is blocking another boy in a blue uniform," CLIP may offer a limited output like "football helmet, desktop computer, swing, scoreboard, shield."

   - **Solution**: A potential improvement could involve extracting adjectives and verbs from the Flicker 8K captions to enrich the information CLIP can offer.

     

3. **Challenges with CLIP-GPT-3 Integration**:

   - **Issue**: The current method of integrating CLIP with GPT-3 is somewhat static, focusing on a unidirectional information flow. This can lead to GPT-3 receiving either insufficient or excessive details from CLIP, hindering the quality of generated captions.
   - **Solution**: Incorporating a back-querying mechanism could be instrumental. By allowing iterative feedback between the models, GPT-3 could refine its queries to CLIP, ensuring it garners the most relevant information for precise caption generation.



