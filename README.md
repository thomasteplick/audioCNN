<h3>Audio Recognition/Classification using a Convolutional Neural Network with the Back-Propagation Algorithm</h3>
<hr>
This program is a web application written in Go that makes extensive use of the html/template package.
Navigate to the C:\Users\your-name\ConvNeuralNetworkAudioRecognition\src\cnn\ directory and issue "go run audiocnn.go" to
start the Convolutional Neural Network server. In a web browser enter http://127.0.0.1:8080/audioCNN
in the address bar.  There are two phases of operation:  the training phase and the testing phase.  During the training
phase, examples consisting of wav audio and the desired class are supplied to the network.  The wav audio files are submitted
to Fourier Analysis using the Fast Fourier Transform (FFT) and the Spectral Power Density in dB is created.  The dB versus frequency plots
are 300x300 Power Spectral Density (PSD) two-dimensional graphs that are converted to Go image structures and inserted into two-dimensional matrices.
The network itself is a directed graph consisting of an input layer of the images, a hidden layer of 30 feature maps, and
a flattened output layer of nodes. The 30 feature map outputs are 30x30 planar neurons, they are pooled using 2x averaging, and 
serialized into a 30*15*15+1 one-dimensional array.  The feature maps of the network are connected by weighted
links.  The network is fully connected.  This means that every node is connected to its immediately adjacent neighbor node.  The weights are trained
by first propagating the inputs forward, layer by layer, to the output layer of nodes.  The output layer of nodes finds the
difference between the desired and its output and back propagates the errors to the input layer.  The hidden and input layer
weights are assigned “credit” for the errors by using the chain rule of differential calculus.  Each neuron consists of a
linear combiner and an activation function.  This program uses the hyperbolic tangent function to serve as the activation function.
This function is non-linear and differentiable and limits its output to be between -1 and 1.  <b>The purpose of this program is to classify an
audio waveform stored in a wav audio file</b>.
The user selects the CNN training parameters:
<li>Epochs</li>
<li>Learning Rate</li>
<br />
<p>
The <i>Learning Rate</i> is between .01 and .00001.  Each <i>Epoch</i> consists of 16 <i>Training Examples</i>.  
One training example is a Power Spectral Density 300x300 image and the desired class (0, 1,…, 15).  There are 16 images and therefore 16 classes.
The 300x300 are converted to 1 or -1 integers that represent the PSD graph.
The 1 represents black  and the -1 represents white.  The WAV audio files were produced using the Generate Audio page.  The user selects the SNR in
dB (0-50).  The program generates 16 wav audio files consisting of 8,192 16-bit samples of 5-8 sinusoids using a sampling frequency of 4,000 Hz (64kbps).  This
produces a two-second wav audio file.  The sinsuoids are between 120 and 1920 Hz, at multiples of 120.  Since the sampling rate is 4,000 Hz, the Nyquist frequency 
is 2,000 Hz.
</p>
<p>
When the <i>Submit</i> button on the MLP Training Parameters form is clicked, the weights in the network are trained
and the Learning Curve (mean-square error (MSE) vs epoch) is graphed.  As can be seen in the screen shots below, there is significant variance over the ensemble,
but it eventually settles down after about 100 epochs. An epoch is the forward and backward propagation of all the 16 training samples.
</p>
<p>
When the <i>Test</i> link is clicked, 16 examples are supplied to the MLP.  It classifies the images.
The test results are tabulated and the actual images are graphed from the png files that were supplied to the CNN.
It takes some trial-and-error with the MLP Training Parameters to reduce the MSE to zero.  It is possible to a specify a 
more complex MLP than necessary and not get good results.  For example, using more hidden layers, a greater layer depth,
or over training with more examples than necessary may be detrimental to the MLP.  Clicking the <i>Train</i> link starts a new training
phase and the MLP Training Parameters must be entered again.
</p>
<h4>Details</h4>
<p>
 The Convolutional Neural Network consists of Feature Maps which are two-dimensional
arrays (planes) of neurons that are arranged in layers.  Each Feature Map has a
10x10 filter or kernel (100 weights) that is used to perform a convolution with previous
layer Feature Maps.  The stride or displacement of each convolution is ten
Padding is used so that the convolution produces the same size output Feature Map as
the input Feature Map.  The input layer is the 300x300 image of the PSD.  There is
one hidden layer, the hidden layer consists of a convolution operation and
a downsample operation.  The downsample operation reduces the Feature Map width and
height by a factor of two.  The last hidden layer is flattened and fully connected
to the output layer.  The hidden layer Feature Maps depths and sizes are as follows:
32@32x32 (32*8 weights).  Each filter in the Feature Maps has a bias input of 1 and weight.
The output layer also has a bias input of 1 and weight.  The output layer is 4@1X1.
This architecture can classify 2^4 = 16 images.  The flattened fully-connected layer
between the last hidden layer and the output layer is 32*16*16 + 1 = 8193 neurons.
These are fully connected to the output layer consisting of the four neurons.
This will require 4*8193 weights. 
</p>

<b>Audio Recognition Learning Curve, MSE vs Epoch, One Hidden Layer, 30 feature maps, 30x30 neurons average downsampled 2x, flattened output layer
<b>0 SNR, Learning Rate .001, Epochs 200
![image](https://github.com/user-attachments/assets/c8029ef4-860a-40ad-82c4-4be57129ffd3)

<b>Audio Recognition Test Results, One Hidden Layer, 30 feature maps, 30x30 neurons average downsampled 2x, flattened output layer</b>
<b>0 SNR, Learning Rate .001, Epochs 200
![image](https://github.com/user-attachments/assets/9720cb82-78ce-4102-b4f4-092385e66073)




