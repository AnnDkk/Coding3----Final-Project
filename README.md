# Coding3----Final-Project
## Electronic kaleidoscope —— by An Dong

<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/%E5%9B%BE%E5%B1%82%201.png" width="800" height="500">
</div>

# Introduction
The Deep Dream model cleverly uses the machine learning principle of overprocessing the identified features, which attracted me deeply. In pursuing machine learning with higher accuracy, there is a model and algorithm that can see its special. And people through this "special" produce interesting arts which show the creativity of artificial intelligence. After my research, I generated a lot of artwork that I worked with AI by understanding and using the various layers and channels in the model. And this triggered my new inspiration for the project.

# Work File
### Video: 


### Procedural code: 
Test1: https://colab.research.google.com/drive/1CRzDCmgjTfhwpau5XA4qZACguivOIxDA#scrollTo=d3g4GwmAfZTK
</div>
Test2: https://colab.research.google.com/drive/1O34vz-9Z71iF0cRzrRtw-4VT8Yxfph0d?hl=zh-cn#scrollTo=pw-1UIEYlprW

### Result code: 
Electronic kaleidoscope: https://colab.research.google.com/drive/1in-DDdF16WzmHUhmmtnaedY4hs2henkK#scrollTo=zE77ZZiAex0I

# Research on Deep Dream Model
According to the article <Inceptionism: Going Deeper into Neural Networks> [3] , I learned about the algorithm used in Deep Dream:
</div> 
Traditional deep learning uses gradient descent to minimize loss. However, this model uses the principle of gradient ascent to process images. It tries to maximize the expression of the identified single feature by increasing the activation value, so as to visually express the characteristics and styles of the picture seen by the machine at each level.

</div>
</div> 

<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/%E8%B5%84%E6%BA%90%201.png" width="800" height="300">
</div>
Deep Dream mainly uses the Inceptionism model. In the paper, they used random noise and augmented inputs in a specific way to show what the neural network thinks some objects look like.
Even though these outputs can be regulated by people rewriting parameters and code. But I still think it's a cool expression! It might be a possible source of some art! So I made a first attempt in the code: to represent the features identified by some layers in InceptionismV3 with random noise.
</div>

First Test Code: 
https://colab.research.google.com/drive/1CRzDCmgjTfhwpau5XA4qZACguivOIxDA#scrollTo=d3g4GwmAfZTK (Reference[2])
</div>

I tested the feature representation of the following layers and different channels within them with random noise.
</div>

<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/2.png" width="800" height="350">
</div> 

The preceding legend shows only a few interesting images. According to the above test results, I found that each channel in these established layers represents a recognition feature. Their graphical characteristics don't change much when I run the same channel over and over again. This means that they are multiple channels combined into a layer to comprehensively represent the desired output of the layer module.

</div> 

Then I began to study the official operating model. [1] https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/deepdream.ipynb
> names = ['mixed3', 'mixed5']
> 
> layers = [base_model.get_layer(name).output for name in names]
In this example, it activates two hierarchical features, mixed3 and mixed5. And it also maximizes the loss via gradient ascent to get a new image acting on the imported image in both feature recognition cases.
</div> 
However, since the generated images have too low resolution and the granularity is the same, it is necessary to increase the image size or increase the ratio of the image to produce a better DeepDream image.

</div> 
<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/WechatIMG819.jpeg" width="400" height="300">       <img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/WechatIMG820.jpeg" width="400" height="300">
</div>  

Through comparison, it is found that the process of deep dream is more like a deep style transfer of the image. It's as if a new filter has been applied to the photo, which still extends in the direction of the loaded image and adds an exaggerated dream-like feature.
</div>  
Then I tried different layers and adjusting step and step_size to affect the result. I noticed that as I increased the value of step_size, the image contained the original image would decrease. If it is too large(Left 0.01    Right 0.08), the input layer will completely lose the effect of the original photo.
</div> 
<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/WechatIMG821.jpeg" width="400" height="300">       <img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/WechatIMG822.jpeg" width="400" height="300">
</div>  

</div>  

## Inspiration
I was pleasantly surprised when I watched [4] 's video! He used the model on a short video of a supermarket visit. He showed what it would be like to see the real world with a "DeepDream" filter.

<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/WechatIMG823.png" width="400" height="300">       <img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/WechatIMG824.png" width="400" height="300">
</div>  


### This wonderful view of the surrounding landscape makes me think of " kaleidoscope "


# Project Production

## Concepts
Based on the previous study and research, I want to enlarge the recognition information of machine learning in each channel more visually. After my test and research, I found that the output results can be maximized by adjusting the parameter information in a separate channel. And the sense of repetition is like a prism in a kaleidoscope. Perhaps I could simply modify the model to achieve an electronic kaleidoscope effect.


## Code
I found the images of mixed1 and mixed 4 to be more interesting, so I modified the code to increase the random noise image, so as to better show the details of the image features.

</div> 

> img_noise = np.random.uniform(size=(300,300,3)) + 100.0
> 
> img_noise = img_noise.astype(np.float32)
> 
> show_image(normalize_image(img_noise))

</div>  

</div>  

By magnifying the noise plot and changing the stesp_size values in the statement, I got the following interesting output:

> def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01, 
                                octaves=range(-2,3), octave_scale=1.3):
> 
> dream_img = render_deepdream(dream_model,img,steps=100,step_size=0.01)


<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/3.png" width="800" height="600">
</div> 


## Challenge

After recording the interesting model channels a new problem arises, how to load images with only a single input processing information of a single channel in a hierarchy module.
</div> 
Original Code:

> names = ['mixed1', 'mixed4']
> 
> layers = [base_model.get_layer(name).output for name in names]
> 
> dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

I started by simply saying that output is changed to the layer channel of my choice. But the code keeps getting errors: ValueError:  Output tensors of a Functional model must be the output of a TensorFlow `Layer` (thus holding past layer metadata).  Found: 79
</div> 


> name = 'mixed5'  # Layer name
> 
> channel_index = 74  # Channel index
> 
> dream_model = tf.keras.Model(inputs=base_model.input, outputs = channel_index)

I searched for users who had this problem, and most of them were determined not to have the channel information sliced and stored in the identified hierarchy, so they could only retrieve the number 79. I made many changes and even asked chatgpt to make changes to this code and it still failed. I eventually did this through a combination of suggested changes.

</div> 

The modified final code:

> name = 'mixed5'  # Layer name
> 
> channel_index = 74  # Channel index
> 
> layer_output = base_model.get_layer(name).output
> 
> selected_channel = layer_output[:, :, :, channel_index]
> 
> dream_model = tf.keras.Model(inputs=base_model.input, outputs=selected_channel)

<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/41.png" width="800" height="400">
</div> 




## Test
After the code worked I tried it out with a new model and image. I find that when I choose a better guide image and adjust the step based on the result, I get better work.

<div align=center>
<img src="https://github.com/AnnDkk/Coding3----Final-Project/blob/main/C3image/5.png" width="800" height="300">
</div> 
The model learns the colour and graphical logic of the loaded image and adds the features of the model channel to achieve a brand-new artwork. Therefore, I believe that the subjective choice of pictures should also be part of art creation. People and machines create new works of art together, just like people holding the same kaleidoscope to look at different scenes will get completely different visual feast!


# Achievements



# Reference
### Code：
[1] https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/deepdream.ipynb
</div>

[2] https://blog.csdn.net/tangkcc/article/details/121453660
</div>


### Article/Viedo：
[3] https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
</div>

[4] https://www.youtube.com/watch?v=DgPaCWJL7XI
</div>

[5]
</div>

[4]











