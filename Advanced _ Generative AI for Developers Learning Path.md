### Introduction to Image Generation
Image Model families
#### Variational AutoEncoders (VAEs)
- Encode images to a compressed size, then decode them back to the original size while learning the distribution of the data
#### Generative Adversarial Networks (GANs)
- Pit two neural networks against each other
- One NN (Generator) generates an image
- The other NN (Discriminator) predicts if the image is real or fake
- Overtime, the discriminator and generator both get better
#### Auto-regressive Models
Generate images by treating images as sequence of pixels
#### Diffusion Models
- Draw inspiration from physics , specially thermodynamics
- Widely used in today's technologies
##### There are a number of different use cases:
###### Unconditioned
- No additional input or instructions
- Can be trained on images of a specific thing to generate new images of that thing
- e,g: Human faces, Super resolution
###### Conditioned
- Text to image ("Mona Lisa with cat face")
- Image editing ("Remove woman from the picture")
- Text-guided image to image ("Disco dancer with colorful lights")
##### Process
1. Destroy structure in a data distribution through iterative forward diffusion process
2. Learn a reverse diffusion process that restores structure in data
	 ![[Pasted image 20240715193806.png]]
3. **Forward Diffusion**
	- Start the forward diffusion process, go from $X_0$ to $X_T$ iteratively, to add more and more noise to the image. $X_T$ is pure noise
	![[Pasted image 20240715200747.png]]
4.  **Training model**
	- We have our initial image $X$ and we sample at timestamp $T$ to create a noisy image
	- We train a denoising model to predict the noise
	- This model is trained to minimize the difference between predicted noise and the actual noise
	- This model is able to remove noise from real images
	![[Pasted image 20240715214202.png]]
5. **Generating an image**
	- To generate an image, we start with pure noise and send it through our denoising model
	- We can take the predicted noise and subtract from the initial noise
	- The model learns from the real data distribution it has seen and sample from those learned distribution to generate new novel images
	![[Pasted image 20240715214736.png]]
### Attention Mechanism
A mechanism that allows a neural network to focus on a specific part of input sequence
- This is done by assigning different weights to different parts of the input sequence
- The most important parts receive the highest weights
- The weighted sum of the input sequence is then used as the input to the next layer of the neural network
##### Traditional Encoder Decoder Neural Network
- The model takes one word at a time as input
- Updates the hidden state
- Passes it on to the next timestamp
- Only the last timestamp is passed on to the decoder for processing
![[Pasted image 20240715221309.png]]
##### Attention model
- Instead of just passing the final hidden state to decoder, the encoder passes all hidden states from each timestamp
- Gives decoder more context beyond the final state
- The decoder uses all the data from hidden states to process
**An attention model takes an extra step before producing output**
- To focus on the most relevant parts of input, the model does the following
1. Looks at the set of encoder hidden states that it received. Each encoder hidden state is associated with certain word in the input sequence
2. Gives each hidden state a score
3. Multiplies each hidden state by its softmax score
	![[Pasted image 20240715222028.png]]
### Encoder-Decoder Architecture
- Sequence to sequence architecture
- Takes as input a sequence of inputs and outputs a sequence
- Consumes sequences and spits out sequences
#### It has two stages
- Both can be implemented with different internal architectures
- The internal architecture can be RNN or transformer block
![[Pasted image 20240716122712.png]]
##### Encoding stage
- Produces vector representation of the input sequence
- RNN encoder takes each token in the input sequence, one at a time, and produces a state representing this token as well as all the previously ingested tokens
- This state is used in next step as input along with the next token to produce the next state
- Once all input tokens are ingested into RNN, an output vector is produced representing full input sequence
##### Decoding stage
- Creates the sequence output
- Takes the vector representation of the input sequence and produces output sentence from that representation
- RNN decoder does this in steps, decoding the output one token at a time, using the current state and already decoded tokens
#### Training
- We need a dataset of input/output pairs that we want our model to imitate
- Feed this dataset into model to correct its weights on the basis of error
- We need a collection of input and output texts
- We'll feed the source to the encoder and compute the error
- ***The decoder also needs its input at training time***
- The decoder needs correct previous token as input to generate the next token
- This method of training is called ***Teacher Forcing***
- This means that we will have to prepare ***two input sequences** in our code
	1. The original input fed into the encoder
	2. The original input shifted to the left fed into the decoder
- The decoder only generates a probability at each step that a token is the next one
- We will have to select a word using these probabilities
	- There are ***several*** approaches for this
	1. ***Greedy Search***, the simplest one, generates the token with highest probability
	2. ***Beam Search***, use the probabilities generated by decoder to evaluate the probability of sentence chunks rather than individual words

![[Pasted image 20240716124345.png]]

#### Serving
- If we want to generate a new response, we will need to feed encoder representation of our input into the decoder along with a special token like "Go"
- This is what happens under the hood
	1. The start token needs to be represented by a vector using ***embedding layer***
	2. The ***recurrent layer*** will update the previous state produced by encoder into a new state
	3. This state will be passed to a dense softmax layer to produce the word probabilities
	4. Finally a word is selected using any approach (Greedy or Beam search etc.)
	5. This process will be repeated until all tokens have been generated
	![[Pasted image 20240716125334.png]]
	- The modern LLMs are based on more complex ***Attention Mechanism***
### Transformer Models and BERT Model
A transformer is an Encoder-Decoder model that uses the Attention Mechanism
- Can take advantage of data parallelization
- Can process large amounts of data at the same time due to its architecture
![[Pasted image 20240717131848.png]]
- Attention Mechanism helps improve performance of ***machine translation*** applications
- Transformer models were built using Attention Mechanism at the core
![[Pasted image 20240717132156.png]]
#### Problems with Pre-Transformer Models
- Models were able to represent words as vectors but lacked context
- The usage of words changes with context
#### Structure
A Transformer model consists of **Encoders** and **Decoders**
![[Pasted image 20240717134138.png]]
##### Encoding Component
A stack of encoders of same number (a hyper-parameter)
##### Encoder
Encodes the input sequence and passes it to the decoder
- Encoders are all identical in structure but with different weights
- Each encoder can be broken down into two **sub layers**
	1. ***Self Attention***
		- The input of the encoder layer first flows through self attention layer
		- Helps the encoder look at the relevant parts of the word as it encodes the central word in the input sentence
		- The input is broken into ***Query***, ***Key*** and ***Value*** vectors
		- These vectors are computed using weights that transformer learns during training
		![[Pasted image 20240717140611.png]]
		- All of these computations happen in parallel in the form of matrix computation
		![[Pasted image 20240717140916.png]]
		- Next, multiply each vector by ***softmax*** score and sum them up
		- This helps keep intact the values of words that we want to focus on and leave out irrelevant words by multiplying them by tiny numbers
		![[Pasted image 20240717141226.png]]
		- Next, we sum up the weighted ***Value*** vectors, which is the output of self-attention layer for the first word
		![[Pasted image 20240717141433.png]]
		![[Pasted image 20240717142056.png]]
	2. ***Feedforward***
		- The output of self-attention layer is fed into Feedforward neural network
		- The exact same FF NN is independently applied to each position
- After embedding the words in the input sequence, each of the embedding vectors flows through the two layers of the encoder
- The word at each position passes through self-attention process
- Then it passes through feedforward neural network, the exact same network with each vector flowing through it separately
![[Pasted image 20240717140133.png]]
- Dependencies exist between these paths in self-attention layer
- However, these dependencies do not exist in the feedforward
##### Decoder
Decodes the information for relevant task
- Decoder has both the self-attention and feedforward layer
- Between these two is the encoder-decoder attention layer that helps the decoder focus on relevant parts of input sentence
### Create Image Captioning Models
![[Pasted image 20240717143100.png]]

### Responsible AI for Developers: Fairness & Bias
See [[Responsible AI]]
#### Bias
- When training a model be aware of common human biases that can manifest in you data
- You need to take proactive actions against such biases
##### Types of Bias
1. ***Reporting Bias***
	- Frequency of events, properties, outcomes etc. does not match their real-world frequency
	- This happens because people document circumstances that are memorable or unusual
	- **EXAMPLE**: A sentiment-analysis model is trained to predict whether book reviews are positive or negative based on a corpus of user submissions to a popular website. The majority of reviews in the training data set reflect extreme opinions (reviewers who either loved or hated a book), because people were less likely to submit a review of a book if they did not respond to it strongly. As a result, the model is less able to correctly predict sentiment of reviews that use more subtle language to describe a book.
2. ***Automation Bias***
	- A tendency to favor results generated by automated systems over non-automated systems
	- Disregards the errors rates of each
	- **EXAMPLE**: Software engineers working for a sprocket manufacturer were eager to deploy the new "groundbreaking" model they trained to identify tooth defects, until the factory supervisor pointed out that the model's precision and recall rates were both 15% lower than those of human inspectors.
3. ***Selection Bias***
	- Chosen examples do not reflect real world distribution
	- **Forms of Selection Bias**
		1. **Coverage Bias**
			- Data is not selected in a representative way
			- **EXAMPLE**: A model is trained to predict future sales of a new product based on phone surveys conducted with a sample of consumers who bought the product. Consumers who instead opted to buy a competing product were not surveyed, and as a result, this group of people was not represented in the training data.
		2. **Non-response Bias/Participation Bias**
			- Data ends up being unrepresentative due to participation gaps in data-collection process
			- **EXAMPLE**: A model is trained to predict future sales of a new product based on phone surveys conducted with a sample of consumers who bought the product and with a sample of consumers who bought a competing product. Consumers who bought the competing product were 80% more likely to refuse to complete the survey, and their data was underrepresented in the sample.
		3. **Sampling Bias**
			- Proper randomization is not used during data collection
			- **EXAMPLE**: A model is trained to predict future sales of a new product based on phone surveys conducted with a sample of consumers who bought the product and with a sample of consumers who bought a competing product. Instead of randomly targeting consumers, the surveyor chose the first 200 consumers that responded to an email, who might have been more enthusiastic about the product than average purchasers.
4. ***Group Attribution Bias***
	- Generalizing what is true of an individual to an entire group to which they belong
	- **Sub-types**
		1. **In-group Bias**
			- Preferring the group you belong to, or share similar characteristics
			- **EXAMPLE**: Two engineers training a resume-screening model for software developers are predisposed to believe that applicants who attended the same computer-science academy as they both did are more qualified for the role.
		2. **Out-group Homogeneity Bias**
			- Stereotyping individuals of group(s) you do not belong to
			- **EXAMPLE**: Two engineers training a resume-screening model for software developers are predisposed to believe that all applicants who did not attend a computer-science academy do not have sufficient expertise for the role.
5. ***Implicit Bias***
	- Making assumptions based on one's own mental models and personal experiences
	- **EXAMPLE**: An engineer training a gesture-recognition model uses a head shake as a feature to indicate a person is communicating the word "no." However, in some regions of the world, a head shake actually signifies "yes."
### MLOps for Gen AI
- A set of practices, techniques and technologies that expand MLOps for operationalizing generative AI applications
#### Traditional ML Workflow
![[Pasted image 20240722135935.png]]
#### MLOps Workflow for GenAI
- Instead of creating a new model from scratch, we tend to **discover** and **fine tune** already existing LLMs
	- We can start to **prompt and predictions** using NLP
- Next, we **train** and serve our **model**. However, there are two new phases
	1. ***Customizing and Tuning***
		- Instead of creating models from scratch, GenAI focuses on customizing and fine tuning models for specific tasks
		- Tailoring powerful models to our specific needs
	2. ***Curated Data***
		- GenAI depends on curated datasets
		![[Pasted image 20240722141246.png]]