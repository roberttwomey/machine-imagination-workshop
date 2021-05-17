# Machine Imagination: Text to Image Generation with Neural Networks

U.Chicago [Digital Media Workshop](https://voices.uchicago.edu/digitalmedia/) and [Poetry and Poetics Workshop](https://voices.uchicago.edu/poetryandpoetics/) | 4-5:30pm CT, May 17, 2021

Robert Twomey, Ph.D. | [roberttwomey.com](roberttwomey.com)

---

# Description

With recent advancements in machine learning techniques, researchers have demonstrated remarkable achievements in image synthesis (BigGAN, StyleGAN), textual understanding (GPT-3), and other areas of text and image manipulation. This hands-on workshop introduces state-of-the-art techniques for text-to-image translation, where textual prompts are used to guide the generation of visual imagery. Participants will gain experience with Open AI's CLIP network and Google's BigGAN, using free Google Colab notebooks which they can apply to their own work after the event. We will discuss other relationships between text and image in art and literature; consider the strengths and limitations of these new techniques; and relate these computational processes to human language, perception, and visual expression and imagination. __Please bring a text you would like to experiment with!__

# Schedule

|    Time    | Activity |
|------------|----|
| 4:00	| Introductions; Open up Google colab; Introduction to Neural Nets, Generative Image (GANs), Generative Text (Transformers). |
| 4:10	| Hands on with CoLab notebook: CLIP + BigGAN + CMA-ES; Talk about format of textual "prompts"/inputs; Explore visual outputs. |
| 4:40	| Check in on results. Participants informally share work with group; Q&A about challenges/techniques. Participants continue working. |
| 5:00	| Hands on with CoLab: Interpolation and latent walks. |
| 5:10	| Discussion, Future Directions | 
| 5:30  | End |

# Notebooks

Click on the links below to open the corresponding notebooks in google colab.

1. Text to Image Generation with BigGAN and CLIP - [text_to_image_BiGGAN_CLIP.ipynb](https://colab.research.google.com/github/roberttwomey/machine-imagination-workshop/blob/main/text_to_image_BigGAN_CLIP.ipynb)
2. Generate latent interpolations - [generate_from_stored.ipynb](https://colab.research.google.com/github/roberttwomey/machine-imagination-workshop/blob/main/generate_from_stored.ipynb)
3. Batch process textual prompts - text_to_image_batch.ipynb (not yet implemented on colab)

# Discussion

- How do words specify/suggest/evoke images? 
- What do you see when you read? Are some texts more or less imagistic?
- How can we use this artificial machine imagination to understand our human visual imagination? 
- How might you incorporate these techniques into our creative production or scholarship? 
- What would it mean to diversify machine imagination? |

# References
- Google Deep Mind BigGAN, [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://deepmind.com/research/publications/large-scale-gan-training-high-fidelity-natural-image-synthesis), 2018
- NVIDIA StyleGAN2, [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), 2019
- OpenAI GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020
- OpenAI CLIP: [Connecting Text and Image](https://openai.com/blog/clip/), 2021
- OpenAI DALL-E: ([link](https://openai.com/blog/dall-e/))
  - Particularly look at the interactive examples. This will give you a sense of the kind of technique we will explore during the workshop.
- Reddit [r/MachineLearning thread on CLIP-related text-to-image notebooks](https://www.reddit.com/r/MachineLearning/comments/ldc6oc/p_list_of_sitesprogramsprojects_that_use_openais/)


<!-- # Leftovers
- What is a GAN (Generative Adversarial Network)? [TK Article on GANs]
 - photorealistic face generation [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/)
- How do computers understand/generate text? [TK]
  - excerpts from Kenric Allado-McDowell's Pharmako-AI
-->
