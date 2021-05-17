# Machine Imagination: Text to Image Generation with Neural Networks

U.Chicago [Digital Media Workshop](https://voices.uchicago.edu/digitalmedia/) and [Poetry and Poetics Workshop](https://voices.uchicago.edu/poetryandpoetics/) | 4-5:30pm CT, May 17, 2021

Robert Twomey, Ph.D. | [roberttwomey.com](roberttwomey.com)

---

## Description

With recent advancements in machine learning techniques, researchers have demonstrated remarkable achievements in image synthesis (BigGAN, StyleGAN), textual understanding (GPT-3), and other areas of text and image manipulation. This hands-on workshop introduces state-of-the-art techniques for text-to-image translation, where textual prompts are used to guide the generation of visual imagery. Participants will gain experience with Open AI's CLIP network and Google's BigGAN, using free Google Colab notebooks which they can apply to their own work after the event. We will discuss other relationships between text and image in art and literature; consider the strengths and limitations of these new techniques; and relate these computational processes to human language, perception, and visual expression and imagination. __Please bring a text you would like to experiment with!__

## Schedule

|    Time    | Activity |
|------------|----|
| 4:00	| Introductions; Open up Google colab; Introduction to Neural Nets, Generative Adversarial Networks (GANs), Generative Text (Transformers). |
| 4:10	| Hands on with CoLab notebook: CLIP + BigGAN + CMA-ES; Talk about format of textual "prompts"/inputs; Explore visual outputs. |
| 4:40	| Check in on results. Participants informally share work with group; Q&A about challenges/techniques. Participants continue working. |
| 5:00	| Hands on with CoLab: Interpolation and latent walks. |
| 5:10	| Discussion, Future Directions | 
| 5:30  | End |

## Notebooks

Click on the links below to open the corresponding notebooks in google colab.

1. BigGAN - [BigGAN_handson.ipynb](https://github.com/roberttwomey/machine-imagination-workshop/blob/main/BigGAN_handson.ipynb)
2. Text to Image Generation with BigGAN and CLIP - [text_to_image_BiGGAN_CLIP.ipynb](https://colab.research.google.com/github/roberttwomey/machine-imagination-workshop/blob/main/text_to_image_BigGAN_CLIP.ipynb)
3. Generate latent interpolations - [generate_from_stored.ipynb](https://colab.research.google.com/github/roberttwomey/machine-imagination-workshop/blob/main/generate_from_stored.ipynb)
4. Batch process textual prompts - text_to_image_batch.ipynb (not yet implemented on colab)

## Discussion

- How do words specify/suggest/evoke images? 
- What do you see when you read? Are some texts more or less imagistic?
- How can we use this artificial machine imagination to understand our human visual imagination? 
- How might you incorporate these techniques into our creative production or scholarship? 
- What would it mean to diversify machine imagination? |

## References
- Google Deep Mind BigGAN, [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://deepmind.com/research/publications/large-scale-gan-training-high-fidelity-natural-image-synthesis), 2018
  - see the BigGAN hands-on notebook above to get a sense for image generation with BigGAN, noise vectors, truncation, and latent interpolation. 
- NVIDIA StyleGAN2, [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), 2019
  - see for example [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/), a photorealistic face generator with StyleGAN2
- OpenAI GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020
  - see Kenric Allado-McDowell's [Pharmako-AI](https://ignota.org/products/pharmako-ai) for an example a book written with GPT-3.
- OpenAI [CLIP: Connecting Text and Image](https://openai.com/blog/clip/), 2021
- OpenAI [DALL-E: Creating Images from Text](https://openai.com/blog/dall-e/), 2021
  - the interactive examples on this page will give you a sense of the kind of technique we will explore during the workshop.
- Good [list of CLIP-related to text-to-image notebooks on Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/ldc6oc/p_list_of_sitesprogramsprojects_that_use_openais/)

## Networks

__Generative Adversarial Networks (GANs)__

<!--![image](https://user-images.githubusercontent.com/1598545/118530742-d74a1300-b6f9-11eb-9743-6d87c96961a3.png)-->
<!-- cropped ![image](https://user-images.githubusercontent.com/1598545/118531573-d5348400-b6fa-11eb-8f53-a324929ef48c.png)-->
<img width="600" alt="GAN diagram with generator and discriminator" src="https://user-images.githubusercontent.com/1598545/118531573-d5348400-b6fa-11eb-8f53-a324929ef48c.png">

A Generative Adversarial Network (GAN) is a kind of generative model. The basic idea is to set up a game between two players (game theory). The Generator creates samples that resemble the input dataset. The Discriminator evaluates samples to determine if they are real or fake (binary classifier). We can think of the generator as being like a counterfeiter, trying to make fake money, and the discriminator as being like police, trying to allow legitimate money and catch counterfeit money. To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from genuine money, and the generator network must learn to create samples that are drawn from the same distribution as the training data. (adversarial) Both networks are trained simultaneously.

Ian Goodfellow introduced the architecture in __Generative Adversarial Nets__, Goodfellow et. al (2014) https://arxiv.org/pdf/1406.2661.pdf

__BigGAN__

<!-- ![image](https://user-images.githubusercontent.com/1598545/118533146-8daef780-b6fc-11eb-8f4a-91b205fb65b5.png)-->
<img width="600" alt="samples from BigGAN" src="https://user-images.githubusercontent.com/1598545/118533146-8daef780-b6fc-11eb-8f4a-91b205fb65b5.png">

BigGAN (2018) set a standard for high resolution, high fidelity image synthesis in 2018. It contained four times as many parameters and eight times the batch size of previous models, and synthesized a state of the art 512 x 512 pixel image across [1000 different classes](https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt) from [Imagenet](https://www.image-net.org/). It was also prohibitively expensive to train! Thankfully Google/Google Brain has released a number of pretrained models for us to explore. Read the paper here https://arxiv.org/abs/1809.11096.

__CLIP__

<!--![image](https://user-images.githubusercontent.com/1598545/118530808-ee890080-b6f9-11eb-8a49-1e1e73097792.png)-->
<img width="600" alt="CLIP diagram" src="https://user-images.githubusercontent.com/1598545/118530808-ee890080-b6f9-11eb-8a49-1e1e73097792.png">

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision. (from https://github.com/openai/CLIP)
