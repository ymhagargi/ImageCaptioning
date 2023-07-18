# Image Captioning on Flickr8k dataset

This project aims to develop an image captioning system using an Encoder-Decoder architecture with Attention. The main objective is to generate descriptive and contextually relevant captions for input images. The project consists of three major components: `Encoder`, `Decoder`, and `CNNToRNN`, which work together to accomplish the task.

**1. Encoder:**
   The `Encoder` class utilizes the InceptionV3 pretrained model to extract features from input images. The model's fully connected layer is replaced with a linear layer to obtain image embeddings of a specified size (`embed_size`). The forward method processes the input images through the InceptionV3 model, applies dropout, and ReLU activation to generate the image embeddings.

**2. Decoder:**
   The `Decoder` class is an LSTM-based decoder responsible for generating captions based on the image embeddings from the `Encoder`. The input captions are first embedded using an embedding layer, and then the LSTM processes the embeddings to produce hidden states. The final linear layer is used to obtain output logits for the vocabulary. The forward method takes image embeddings and tokenized captions as inputs and returns the predicted logits.

**3. CNNToRNN:**
   The `CNNToRNN` class combines the `Encoder` and `Decoder` to create the final image captioning model. It facilitates the flow of information from the `Encoder` to the `Decoder`. The forward method takes images and tokenized captions as inputs and returns the predicted logits for the vocabulary.

The overall project workflow involves first extracting features from the input images using the pretrained InceptionV3 model in the `Encoder`. These features are then passed to the `Decoder`, along with the tokenized captions, to generate descriptive captions for the images. The `CNNToRNN` class acts as a bridge between the `Encoder` and `Decoder`, coordinating the image-to-caption generation process.

The model leverages an attention mechanism during caption generation, allowing the `Decoder` to focus on relevant parts of the image while generating each word in the caption. This attention mechanism enhances the model's ability to produce more contextually accurate and visually grounded captions.

The project utilizes PyTorch as the deep learning framework and takes advantage of GPU acceleration (if available) to speed up computations during training and inference. The image captioning model can be trained on a dataset with paired images and corresponding captions. During evaluation, the trained model can generate captions for new input images, demonstrating its ability to describe visual content effectively.
