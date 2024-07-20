/*
Neural Network (nn) using convolutional Feature Map architecture.
This is a web application that uses the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/audioCNN.  There are two phases of
operation:  the training phase and the testing phase.  Epochs consising of
a sequence of examples are used to train the nn.  Each example consists
of a png image and a desired class output.  The nn
itself consists of an input layer of nodes, one or more hidden layers of nodes,
and an output layer of nodes.  The nodes are connected by weighted links.  The
weights are trained by back propagating the output layer errors backward to the
input layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layers.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.

The Convolutional Neural Network cosists of Feature Maps which are two-dimensional
arrays (planes) of neurons that are arranged in layers.  Each Feature Map has a
10x10 filter or kernel (100 weights) that is used to perform a convolution with previous
layer Feature Maps.  The stride or displacement of each convolution is ten.
The input layer is the 300x300 Power Spectral Density image.  There is
one hidden layer and the hidden layer consists of a convolution operation and
a downsample operation.  The downsample operation reduces the Feature Map width and
height by a factor of two.  The last hidden layer is flattened and fully connected
to the output layer.  The hidden layer Feature Maps depths and sizes are as follows:
First Hidden Layer has 30 Feature Maps, each 30x30 neurons.  Each filter in the
Feature Maps has a bias input of 1 and weight.
The output layer also has a bias input of 1 and weight.  The output layer is 4@1X1.
This architecture can classify 2^4 = 16 images.  The flattened fully-connected layer
between the last hidden layer and the output layer is 30*30*30 + 1 = 27001 neurons.
These are fully connected to the output layer consisting of the four neurons.
This will require 4*27001 weights.

This program classifies audio wav files.  An FFT is performed on each 8,000 sample
audio file and the 300x300 image of the Power Spectral Density (PSD) is submitted
to the convolutional neural network.
The CNN is trained and tested on the 16 images.  Each wav audio file consists of 3 to 6
sine waves with Gaussian noise at a specified SNR.  The sampling rate is 4,000 Hz
which produces 2 seconds worth of data.  The samples are 16-bit integers.
The PSD of each file can be viewed in the Audio Generation page or
the Testing page.  The audio wav files can be played in Windows Media Player.

The github.com/go-audio/wav and github.com/go-audio/audio packages are used to generate and process
the audio files.  The github.com/mjibson/go-dsp/fft package is used for generating  the FFT.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
)

const (
	addr                   = "127.0.0.1:8080"                 // http server listen address
	fileTrainingCNN        = "templates/trainingCNN.html"     // html for training CNN
	fileTestingCNN         = "templates/testingCNN.html"      // html for testing CNN
	fileAudioGeneration    = "templates/audioGeneration.html" // html for audio wav file generation
	patternTrainingCNN     = "/audioCNN"                      // http handler for training the CNN
	patternTestingCNN      = "/audioCNNtest"                  // http handler for testing the CNN
	patternAudioGeneration = "/audioGeneration"               // http handler for generating audio wav files
	xlabels                = 11                               // # labels on x axis
	ylabels                = 11                               // # labels on y axis
	fileweights            = "weights.csv"                    // cnn weights
	a                      = 1.7159                           // activation function const
	b                      = 2.0 / 3.0                        // activation function const
	K1                     = b / a
	K2                     = a * a
	dataDir                = "data/"              // directory for the weights and audio
	maxClasses             = 40                   // max number of images to classify
	imgWidth               = 300                  // image width
	imgHeight              = 300                  // image height
	imageSize              = imgWidth * imgHeight // image size
	classes                = 16                   // number of images to classify
	rows                   = 300                  // rows in canvas
	cols                   = 300                  // columns in canvas
	hiddenLayers           = 1                    // number of hidden layers
	kernelDim              = 10                   // kernel dimension, height and width
	stride                 = 10                   // stride for filter forward and backward
	sampleRate             = 4000                 // Hz
	SAMPLES                = 8192                 // audio samples
	twoPi                  = 2.0 * math.Pi        // 2Pi
	bitDepth               = 16                   // audio wav encoder/decoder bit size
)

// Type to contain all the HTML template actions
type PlotT struct {
	Grid         []string  // plotting grid
	Status       string    // status of the plot
	Xlabel       []string  // x-axis labels
	Ylabel       []string  // y-axis labels
	LearningRate string    // size of weight update for each iteration
	Epochs       string    // number of epochs
	TestResults  []Results // tabulated statistics of testing
	TotalCount   string    // Results tabulation
	TotalCorrect string
	FFTSize      string // 8192, 4098, 2048, 1024
	FFTWindow    string // Bartlett, Welch, Hamming, Hanning, Rectangle
}

// Type to hold the minimum and maximum data values of the MSE in the Learning Curve
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// graph node
type Node struct {
	y     float64 // output of this node for forward prop
	delta float64 // local gradient for backward prop
}

// filter or kernel used in convolution on the y or delta
// forward prop filters the y, backward prop filters the delta
type Filter struct {
	wgt     [kernelDim][kernelDim]float64 // kernel weights used in convolution for a Feature Map
	biaswgt float64                       // bias weight whose input is constant 1
	layer   int                           // Feature Map (FM) layer
	n       int                           // nth Feature Map in this layer, one filter per FM
}

type Stats struct {
	correct    []int // % correct classifcation
	classCount []int // #samples in each class
}

// training examples
type Sample struct {
	name    string   // frequencies in Hz contained in the audio file
	desired int      // numerical class of the image
	image   [][]int8 // PSD image 300x300, [-1,1], 1=black, -1=white
}

// Feature Map consists of height, width, and Nodes
type FeatureMap struct {
	h    int // height of Feature Map
	w    int // width of Feature Map
	data [][]Node
}

// Primary data structure for holding the CNN state
type CNN struct {
	plot         *PlotT         // data to be distributed in the HTML template
	Endpoints                   // embedded struct
	link         [][]Filter     // links in the graph which connect the Feature Maps (nodes)
	wgtOutput    []float64      // output flattened weights don't use kernel
	node         [][]Node       // last two flattened layers in the graph are not Feature Maps
	fm           [][]FeatureMap // Feature Maps in the graph, consider them the nodes, [layer, n]
	samples      []Sample
	statistics   Stats
	mse          []float64 // mean square error in output layer per epoch used in Learning Curve
	epochs       int       // number of epochs
	learningRate float64   // learning rate parameter
	desired      []float64 // desired output of the sample
	fftSize      int       // 1024, 2048, 4096, 8192
	fftWindow    string    // one of the winTypes
}

// test statistics that are tabulated in HTML
type Results struct {
	Class   string // int
	Correct string // int      percent correct
	Image   string // image
	Count   string // int      number of training examples in the class
}

// Window function type
type Window func(n int, m int) complex128

// global variables and CNN architecture
var (
	// parse and execution of the html templates
	tmplTrainingCNN     *template.Template
	tmplTestingCNN      *template.Template
	tmplAudioGeneration *template.Template
	// number of Feature Maps in the input + hidden layers
	numFMs [hiddenLayers + 1]int = [hiddenLayers + 1]int{1, 30}
	// Two-dimension sizes of the Feature Maps in the layers
	sizeFMs [hiddenLayers + 1]int = [hiddenLayers + 1]int{300, 15}
	// staging location for pooling of convolution
	clipboard [imgHeight][imgWidth]Node
	// audio wav signal
	audioSamples = make([]float64, SAMPLES)
	winType      = []string{"Bartlett", "Welch", "Hamming", "Hanning", "Rectangle"}
)

// Bartlett window
func bartlett(n int, m int) complex128 {
	real := 1.0 - math.Abs((float64(n)-float64(m))/float64(m))
	return complex(real, 0)
}

// Welch window
func welch(n int, m int) complex128 {
	x := math.Abs((float64(n) - float64(m)) / float64(m))
	real := 1.0 - x*x
	return complex(real, 0)
}

// Hamming window
func hamming(n int, m int) complex128 {
	return complex(.54-.46*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Hanning window
func hanning(n int, m int) complex128 {
	return complex(.5-.5*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Rectangle window
func rectangle(n int, m int) complex128 {
	return 1.0
}

// calculateMSE calculates the MSE at the output layer every epoch
func (cnn *CNN) calculateMSE(epoch int) {
	// loop over the output layer nodes
	var err float64 = 0.0
	outputLayer := len(cnn.node) - 1
	for n := 0; n < len(cnn.node[outputLayer]); n++ {
		// Calculate (desired[n] - cnn.node[L][n].y)^2 and store in cnn.mse[n]
		err = float64(cnn.desired[n]) - cnn.node[outputLayer][n].y
		err2 := err * err
		cnn.mse[epoch] += err2
	}
	cnn.mse[epoch] /= float64(classes)

	// calculate min/max mse
	if cnn.mse[epoch] < cnn.ymin {
		cnn.ymin = cnn.mse[epoch]
	}
	if cnn.mse[epoch] > cnn.ymax {
		cnn.ymax = cnn.mse[epoch]
	}
}

// determineClass determines testing example class given sample number and sample
func (cnn *CNN) determineClass(j int, sample Sample) error {
	// At output layer, classify example, increment class count, %correct

	// convert node outputs to the class; 0.0 is the threshold for sigmoid tanh
	class := 0
	for i, output := range cnn.node[1] {
		if output.y > 0.0 {
			class |= (1 << i)
		}
	}

	// Assign Stats.correct, Stats.classCount
	cnn.statistics.classCount[sample.desired]++
	if class == sample.desired {
		cnn.statistics.correct[class]++
	}

	return nil
}

// class2desired constructs the desired output from the given class
func (cnn *CNN) class2desired(class int) {
	// tranform int to slice of -1 and 1 representing the 0 and 1 bits
	for i := 0; i < len(cnn.desired); i++ {
		if class&1 == 1 {
			cnn.desired[i] = 1
		} else {
			cnn.desired[i] = -1
		}
		class >>= 1
	}
}

// convolve delta and y to update the 10x10 filter with no padding.
func (cnn *CNN) updateFilter(layer int, i1, i2, d1 int) error {
	// conv(Y, Delta) with no padding; complete overlap of Y and Delta.
	// Multiply Conv(Y,Delta)  by learning rate and add to current filter

	// height and width of the Y from previous layer and the upsampled delta
	dim := sizeFMs[layer]
	// Use L for rotating 180 deg
	//L := dim - 1

	sum := 0.0
	for row := 0; row < dim/stride; row++ {
		for col := 0; col < dim/stride; col++ {
			// upsample the average delta in a 2x2 window, this is not efficient
			avg := cnn.fm[layer+1][i1].data[row/2][col/2].delta / 4.0
			// Rotate the kernel 180 deg
			//sum += cnn.fm[layer][i2].data[L-row][L-col].y * avg
			sum += cnn.fm[layer][i2].data[row][col].y * avg
		}
	}
	wgtDelta := sum * cnn.learningRate
	depth := i2*d1 + i1
	// update the weights in the kernel with the same convolution
	for j := 0; j < kernelDim; j++ {
		for i := 0; i < kernelDim; i++ {
			cnn.link[layer][depth].wgt[j][i] += wgtDelta
		}
	}

	return nil
}

// Convolve in the backward propagation direction.
// Filter the local gradients from the downstream layer FMs
func (cnn *CNN) filterB(f *Filter, layer, i1 int) error {
	// Convolve the filter over the delta and don't use padding
	// around the edges.

	// Perform the operations in the clipboard
	data := clipboard

	// height and width of the data that the filter convolves over
	dim := sizeFMs[layer]
	// Rotate the filter in x and y coordinates
	//L := kernelDim - 1
	for row := 0; row < dim; row++ {
		for col := 0; col < dim; col++ {
			sum := 0.0
			curRow := row
			for j := 0; j < kernelDim; j++ {
				curCol := col
				for i := 0; i < kernelDim; i++ {
					sum += f.wgt[j][i] * cnn.fm[layer][i1].data[curRow][curCol].delta
					//sum += f.wgt[L-j][L-i] * cnn.fm[layer][i1].data[curRow][curCol].delta
					curCol++
					if curCol == dim {
						break
					}
				}
				curRow++
				if curRow == dim {
					break
				}
			}
			// save the filtered delta in clipboard
			data[row][col].delta = sum
		}
	}

	// Put the filtered delta back in the FeatureMap.data[row][col].delta
	for row := 0; row < dim; row++ {
		for col := 0; col < dim; col++ {
			cnn.fm[layer][i1].data[row][col].delta = data[row][col].delta
		}
	}

	return nil
}

// convolve in the forward propagation direction
// filter the fm[][].data[][].y from the previous layer FM
func (cnn *CNN) filterF(f *Filter, layer, i1, i2 int) error {
	// Convolve the filter over the Feature Map and don't use
	// padding around the edges.
	// Include the bias weight in the convolution.
	// Compute activation function phi for the convolution output.
	// Downsample by 2 by finding avg over 2x2 window.
	// Put the downsampled FM.data into this layer FM.data.
	// Process the data in the clipboard.

	// Perform the operations in the clipboard
	data := clipboard

	// height and width of the data that the filter convolves over
	dim := sizeFMs[layer]
	for row := 0; row < dim; row += stride {
		for col := 0; col < dim; col += stride {
			// multiply bias weight by constant 1
			sum := f.biaswgt
			curRow := row

			for j := 0; j < kernelDim; j++ {
				curCol := col
				for i := 0; i < kernelDim; i++ {
					sum += f.wgt[j][i] * cnn.fm[layer][i2].data[curRow][curCol].y
					curCol++
				}
				curRow++
			}
			// compute output y = Phi(v) with the activation function
			data[row/stride][col/stride].y = a * math.Tanh(b*sum)
		}
	}

	// Process the data in the clipboard.
	// Downsample by 2 by finding avg over 2x2 window.
	// Put the downsampled FM.data into this layer's FM.data.
	step := 2
	for row := 0; row < dim/stride; row += step {
		row2 := row / 2
		for col := 0; col < dim/stride; col += step {
			col2 := col / 2
			sum := 0.0
			for j := 0; j < step; j++ {
				for i := 0; i < step; i++ {
					sum += data[row+j][col+i].y
				}
			}
			// average and propagate to next layer
			cnn.fm[layer+1][i1].data[row2][col2].y = sum / 4.0
		}
	}

	return nil
}

func (cnn *CNN) propagateForward(samp Sample, epoch int) error {
	// Assign sample to input layer FM
	layer := 0
	for j := 0; j < imgHeight; j++ {
		for i := 0; i < imgWidth; i++ {
			cnn.fm[layer][0].data[j][i].y = float64(samp.image[j][i])
		}
	}

	// calculate desired from the class
	cnn.class2desired(samp.desired)

	// Loop over layers: input + hiddenLayers + output layer
	// input->first hidden, then hidden->hidden,..., then hidden->output
	for layer := 1; layer < len(numFMs); layer++ {
		// Loop over FMs in the layer, d1 is the layer depth of current
		d1 := numFMs[layer]
		for i1 := 0; i1 < d1; i1++ { // this layer loop
			// The network is fully connected.  d2 is the layer depth of previous
			d2 := numFMs[layer-1]
			// filter (convolve) using the kernel in i1
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				// Convolve the previous layer y with the filter connecting
				// this layer to the previous layer.
				err := cnn.filterF(&cnn.link[layer-1][i2*d1+i1], layer-1, i1, i2)
				if err != nil {
					fmt.Printf("filter forward propagate error: %v\n", err.Error())
					return fmt.Errorf("filter forward propagate error: %v", err)
				}
			}
		}
	}

	// Flatten the last FM layer and insert into linear array, 30*15*15=2250
	// Take the downsampled FM from the temp FM
	// node[0][0] is the bias = 1, so skip k = 0
	cnn.node[0][0].y = 1.0
	k := 1
	n := len(numFMs) - 1
	for i := 0; i < numFMs[n]; i++ {
		for row := 0; row < sizeFMs[n]; row++ {
			for col := 0; col < sizeFMs[n]; col++ {
				cnn.node[0][k].y = cnn.fm[n][i].data[row][col].y
				k++
			}
		}
	}

	// Last layers uses flattened fully-connected MLP arrangement.
	// Propagate forward as in MLP
	d1 := len(cnn.node[1])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		// Each FM in previous layer is connected to current FM because
		// the network is fully connected.  d2 is the layer depth of previous
		d2 := len(cnn.node[0])
		// Loop over weights to get v
		v := 0.0
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			v += cnn.wgtOutput[i2*d1+i1] * cnn.node[0][i2].y
		}
		// compute output y = Phi(v) is the logistic or sigmoid function
		cnn.node[1][i1].y = a * math.Tanh(b*v)
	}

	return nil
}

func (cnn *CNN) propagateBackward() error {

	// output layer is different, no bias node, so the indexing is different
	// Loop over nodes in output layer
	d1 := len(cnn.node[1])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		//compute error e=d-Phi(v)
		cnn.node[1][i1].delta = cnn.desired[i1] - cnn.node[1][i1].y
		// Multiply error by this node's Phi'(v) to get local gradient.
		cnn.node[1][i1].delta *= K1 * (K2 - cnn.node[1][i1].y*cnn.node[1][i1].y)
		// Send this node's local gradient to previous layer nodes through corresponding link.
		// Each node in previous layer is connected to current node because the network
		// is fully connected.  d2 is the previous layer depth
		d2 := len(cnn.node[0])
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			cnn.node[0][i2].delta += cnn.wgtOutput[i2*d1+i1] * cnn.node[1][i1].delta
			// Update weight with y and local gradient
			cnn.wgtOutput[i2*d1+i1] +=
				cnn.learningRate * cnn.node[1][i1].delta * cnn.node[0][i2].y

		}
		// Reset this local gradient to zero for next training example
		cnn.node[1][i1].delta = 0.0
	}

	// Insert the flattened output layer local gradients into the last Feature Map
	// Go from linear to planar data.  Upsample the local gradients 2x by inserting
	// avg in elements.  Skip k = 0 because that is the bias delta for the flattened layers.
	k := 1
	n := len(numFMs) - 1
	for i := 0; i < numFMs[n]; i++ {
		for row := 0; row < sizeFMs[n]; row++ {
			for col := 0; col < sizeFMs[n]; col++ {
				cnn.fm[n][i].data[row][col].delta = cnn.node[0][k].delta
				// Reset this local gradient to zero for next training example
				cnn.node[0][k].delta = 0.0
				k++
			}
		}
	}

	// Loop over layers in backward direction, starting at the last hidden layer
	for layer := n; layer > 0; layer-- {
		// Loop over FMs in this layer, d1 is the current layer depth
		d1 := len(cnn.fm[layer])
		for i1 := 0; i1 < d1; i1++ { // this layer loop
			// Multiply deltas propagated from downstream FMs by this node's Phi'(v) to get local gradient.
			for j := range cnn.fm[layer][i1].data {
				for i := range cnn.fm[layer][i1].data[j] {
					cnn.fm[layer][i1].data[j][i].delta *=
						K1 * (K2 - cnn.fm[layer][i1].data[j][i].y*cnn.fm[layer][i1].data[j][i].y)

				}
			}

			// Filter (convolve) this layer's delta and send to previous layers.
			// Each FM in previous layer is connected to current FM because the network
			// is fully connected.  d2 is the previous layer depth
			d2 := len(cnn.fm[layer-1])
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				// convolve the delta with the filter
				err := cnn.filterB(&cnn.link[layer-1][i2*d1+i1], layer, i1)
				if err != nil {
					fmt.Printf("filter backward propagate error: %v\n", err.Error())
					return fmt.Errorf("filter backward propagate error: %v", err)
				}

				// Update filter by convolving y from previous layer and upsampled local gradient
				err = cnn.updateFilter(layer-1, i1, i2, d1)
				if err != nil {
					fmt.Printf("updateFilter backward propagate error: %v\n", err.Error())
					return fmt.Errorf("updateFilter backward propagate error: %v", err)
				}
			}

			// Reset this local gradient to zero for next training example
			for i := range cnn.fm[layer][i1].data {
				for j := range cnn.fm[layer][i1].data[i] {
					cnn.fm[layer][i1].data[i][j].delta = 0.0
				}
			}
		}
	}
	return nil
}

// runEpochs performs forward and backward propagation over each sample
func (cnn *CNN) runEpochs() error {

	// Initialize the Filter 10x10 weights and flattened output weights

	// hidden layer filters, excluding the output layer
	// initialize the Filter wgt randomly, zero mean, normalize by fan-in
	for layer := range cnn.link {
		for n := range cnn.link[layer] {
			cnn.link[layer][n].biaswgt = 2.0 * (rand.Float64() - .5) / (kernelDim * kernelDim)
			for i := range cnn.link[layer][n].wgt {
				for j := range cnn.link[layer][n].wgt[i] {
					cnn.link[layer][n].wgt[i][j] = 2.0 * (rand.Float64() - .5) / (kernelDim * kernelDim)
					//cnn.link[layer][n].wgt[i][j] = rand.Float64() / (kernelDim * kernelDim)
				}
			}
		}
	}

	// output layer links
	for i := range cnn.wgtOutput {
		cnn.wgtOutput[i] = 2.0 * (rand.Float64() - .5) / (kernelDim * kernelDim)
		//cnn.wgtOutput[i] = rand.Float64() / (kernelDim * kernelDim)
	}

	for n := 0; n < cnn.epochs; n++ {
		// Loop over the training examples
		for _, samp := range cnn.samples {
			// Forward Propagation
			err := cnn.propagateForward(samp, n)
			if err != nil {
				return fmt.Errorf("forward propagation error: %s", err.Error())
			}

			// Backward Propagation
			err = cnn.propagateBackward()
			if err != nil {
				return fmt.Errorf("backward propagation error: %s", err.Error())
			}
		}

		// At the end of each epoch, loop over the output nodes and calculate mse
		cnn.calculateMSE(n)

		// Shuffle training exmaples
		rand.Shuffle(len(cnn.samples), func(i, j int) {
			cnn.samples[i], cnn.samples[j] = cnn.samples[j], cnn.samples[i]
		})
	}

	return nil
}

// init parses the html template files
func init() {
	tmplTrainingCNN = template.Must(template.ParseFiles(fileTrainingCNN))
	tmplTestingCNN = template.Must(template.ParseFiles(fileTestingCNN))
	tmplAudioGeneration = template.Must((template.ParseFiles(fileAudioGeneration)))
}

// createExamples creates a slice of training or testing examples
func (cnn *CNN) createExamples() error {
	// Read in audio wav files and create a power spectral density image.
	// Points in the PSD are given values = 1 and are visible in the image (black).
	// Non-visible points (white) are given values = -1.
	files, err := os.ReadDir(dataDir)
	if err != nil {
		fmt.Printf("ReadDir for %s error: %v\n", dataDir, err)
		return fmt.Errorf("ReadDir for %s error %v", dataDir, err.Error())
	}

	var (
		endpoints Endpoints
		N         int       //  complex FFT size
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
	)
	N = cnn.fftSize

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD = make([]float64, N/2)

	// Each audio wav file is a separate class
	class := 0
	// image display convention: 1 means black, -1 means white (background)
	for _, dirEntry := range files {
		name := dirEntry.Name()
		if filepath.Ext(dirEntry.Name()) == ".wav" {
			f, err := os.Open(path.Join(dataDir, name))
			if err != nil {
				fmt.Printf("Open %s error: %v\n", name, err)
				return fmt.Errorf("file Open %s error: %v", name, err.Error())
			}
			defer f.Close()
			// only process classes files
			if class == classes {
				return fmt.Errorf("can only process %v audio wav files", classes)
			}

			dec := wav.NewDecoder(f)
			bufInt := audio.IntBuffer{
				Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
				Data:   make([]int, SAMPLES), SourceBitDepth: bitDepth}
			_, err = dec.PCMBuffer(&bufInt)
			if err != nil {
				fmt.Printf("PCMBuffer error: %v\n", err)
				return fmt.Errorf("PCMBuffer error: %v", err.Error())
			}
			bufFlt := bufInt.AsFloatBuffer()

			// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
			psdMin, psdMax, err := cnn.calculatePSD(bufFlt.Data, PSD)
			if err != nil {
				fmt.Printf("calculatePSD error: %v\n", err)
				return fmt.Errorf("calculatePSD error: %v", err.Error())
			}

			endpoints.xmin = 0.0
			endpoints.xmax = float64(N / 2) // equivalent to Nyquist critical frequency
			endpoints.ymin = psdMin
			endpoints.ymax = psdMax

			// EP means endpoints
			lenEPx := endpoints.xmax - endpoints.xmin
			lenEPy := endpoints.ymax - endpoints.ymin
			prevBin := 0.0
			prevPSD := PSD[0]

			// Calculate scale factors for x and y
			xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
			yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

			// Initialize all points in the grid offline or invisible
			for row := 0; row < imgHeight; row++ {
				for col := 0; col < imgWidth; col++ {
					cnn.samples[class].image[row][col] = -1
				}
			}

			// This previous cell location (row,col) is on the line (visible)
			row := int((endpoints.ymax-PSD[0])*yscale + .5)
			col := int((0.0-endpoints.xmin)*xscale + .5)
			cnn.samples[class].image[row][col] = 1

			// Store the PSD in the plot Grid
			for bin := 1; bin < N/2; bin++ {

				// This current cell location (row,col) is on the line (visible)
				row := int((endpoints.ymax-PSD[bin])*yscale + .5)
				col := int((float64(bin)-endpoints.xmin)*xscale + .5)
				cnn.samples[class].image[row][col] = 1

				// Interpolate the points between previous point and current point;
				// draw a straight line between points.
				lenEdgeBin := math.Abs((float64(bin) - prevBin))
				lenEdgePSD := math.Abs(PSD[bin] - prevPSD)
				ncellsBin := int(float64(cols) * lenEdgeBin / lenEPx) // number of points to interpolate in x-dim
				ncellsPSD := int(float64(rows) * lenEdgePSD / lenEPy) // number of points to interpolate in y-dim
				// Choose the biggest
				ncells := ncellsBin
				if ncellsPSD > ncells {
					ncells = ncellsPSD
				}

				stepBin := float64(float64(bin)-prevBin) / float64(ncells)
				stepPSD := float64(PSD[bin]-prevPSD) / float64(ncells)

				// loop to draw the points
				interpBin := prevBin
				interpPSD := prevPSD
				for i := 0; i < ncells; i++ {
					row := int((endpoints.ymax-interpPSD)*yscale + .5)
					col := int((interpBin-endpoints.xmin)*xscale + .5)
					// This cell location (row,col) is on the line (visible)
					cnn.samples[class].image[row][col] = 1
					interpBin += stepBin
					interpPSD += stepPSD
				}

				// Update the previous point with the current point
				prevBin = float64(bin)
				prevPSD = PSD[bin]

			}

			// save the name of the image without the ext
			cnn.samples[class].name = strings.Split(name, ".")[0]
			// The desired output of the CNN is class
			cnn.samples[class].desired = class

			class++
		}
	}
	fmt.Printf("Read %d png files\n", class)

	return nil
}

// newCNN constructs an CNN instance for training
func newCNN(r *http.Request, epochs int, plot *PlotT) (*CNN, error) {
	// Read the training parameters in the HTML Form

	txt := r.FormValue("learningrate")
	learningRate, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("learningrate float conversion error: %v\n", err)
		return nil, fmt.Errorf("learningrate float conversion error: %s", err.Error())
	}

	fftWindow := r.FormValue("fftwindow")

	txt = r.FormValue("fftsize")
	fftSize, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("fftsize int conversion error: %v\n", err)
		return nil, fmt.Errorf("fftsize int conversion error: %s", err.Error())
	}

	cnn := CNN{
		epochs:       epochs,
		learningRate: learningRate,
		fftWindow:    fftWindow,
		fftSize:      fftSize,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		samples: make([]Sample, classes),
	}

	// construct container for images
	for i := range cnn.samples {
		cnn.samples[i].image = make([][]int8, imgHeight)
		for j := range cnn.samples[i].image {
			cnn.samples[i].image[j] = make([]int8, imgWidth)
		}
	}

	// *********** Links ******************************************************

	// construct links that hold the filters
	cnn.link = make([][]Filter, hiddenLayers)
	// hidden layer filters, excluding the output layer
	// previous layer FM depth
	m := 1
	for i, n := range numFMs[1:] {
		// fully-connected hidden layers
		k := m * n
		cnn.link[i] = make([]Filter, k)
		for j := 0; j < k; j++ {
			cnn.link[i][j] = Filter{n: j, layer: i}
		}
		m = n
	}

	// outer layer uses fully connected MLP using flattened last
	// hidden layer Feature Maps = 30*(15)*(15) = 2250, using downsampled FM
	// added bias weight with constant input = 1
	i := len(numFMs) - 1
	m = numFMs[i]*sizeFMs[i]*sizeFMs[i] + 1
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// output layer links
	cnn.wgtOutput = make([]float64, olnodes*m)

	// ******************* Feature Maps and Nodes ****************************
	// Input and hidden layer Feature Maps, flattened output layer nodes
	cnn.fm = make([][]FeatureMap, len(numFMs))
	cnn.node = make([][]Node, 2)

	//  input and hidden layers
	for i := 0; i < len(numFMs); i++ {
		cnn.fm[i] = make([]FeatureMap, numFMs[i])
		for j := 0; j < numFMs[i]; j++ {
			nodes := make([][]Node, sizeFMs[i])
			for k := 0; k < sizeFMs[i]; k++ {
				nodes[k] = make([]Node, sizeFMs[i])
			}
			cnn.fm[i][j] = FeatureMap{h: sizeFMs[i], w: sizeFMs[i], data: nodes}
		}
	}

	// next to last layer is the flattended last hidden layer
	cnn.node[0] = make([]Node, m)
	// init bias node to 1
	cnn.node[0][0].y = 1.0

	// output layer, which has no bias node
	cnn.node[1] = make([]Node, olnodes)

	// *************************************************************

	// construct desired from classes, binary representation
	cnn.desired = make([]float64, olnodes)

	// mean-square error
	cnn.mse = make([]float64, epochs)

	return &cnn, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (cnn *CNN) gridFillInterp() error {
	var (
		x            float64 = 0.0
		y            float64 = cnn.mse[0]
		prevX, prevY float64
		xscale       float64
		yscale       float64
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (cnn.xmax - cnn.xmin)
	yscale = float64(rows-1) / (cnn.ymax - cnn.ymin)

	cnn.plot.Grid = make([]string, rows*cols)

	// This cell location (row,col) is on the line (visible)
	row := int((cnn.ymax-y)*yscale + .5)
	col := int((x-cnn.xmin)*xscale + .5)
	cnn.plot.Grid[row*cols+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := cnn.ymax - cnn.ymin
	lenEPx := cnn.xmax - cnn.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < cnn.epochs; i++ {
		x++
		// ensemble average of the mse
		y = cnn.mse[i]

		// This cell location (row,col) is on the line (visible)
		row := int((cnn.ymax-y)*yscale + .5)
		col := int((x-cnn.xmin)*xscale + .5)
		cnn.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(float64(cols) * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(float64(rows) * lenEdgeY / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := ncellsX
		if ncellsY > ncells {
			ncells = ncellsY
		}

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((cnn.ymax-interpY)*yscale + .5)
			col := int((interpX-cnn.xmin)*xscale + .5)
			// This cell location (row,col) is on the line (visible)
			cnn.plot.Grid[row*cols+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (cnn *CNN) insertLabels() {
	cnn.plot.Xlabel = make([]string, xlabels)
	cnn.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (cnn.xmax - cnn.xmin) / (xlabels - 1)
	x := cnn.xmin
	// First label is empty for alignment purposes
	for i := range cnn.plot.Xlabel {
		cnn.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (cnn.ymax - cnn.ymin) / (ylabels - 1)
	y := cnn.ymin
	for i := range cnn.plot.Ylabel {
		cnn.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// handleTraining performs forward and backward propagation to calculate the weights
func handleTrainingCNN(w http.ResponseWriter, r *http.Request) {

	var (
		plot PlotT
		cnn  *CNN
	)

	// Get the number of epochs
	txt := r.FormValue("epochs")
	// Need epochs to continue
	if len(txt) > 0 {
		epochs, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Epochs int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Epochs conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create CNN instance to hold state
		cnn, err = newCNN(r, epochs, &plot)
		if err != nil {
			fmt.Printf("newCNN() error: %v\n", err)
			plot.Status = fmt.Sprintf("newCNN() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Create training examples by reading in the encoded characters
		err = cnn.createExamples()
		if err != nil {
			fmt.Printf("createExamples error: %v\n", err)
			plot.Status = fmt.Sprintf("createExamples error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Loop over the Epochs
		err = cnn.runEpochs()
		if err != nil {
			fmt.Printf("runEnsembles() error: %v\n", err)
			plot.Status = fmt.Sprintf("runEnsembles() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put MSE vs Epoch in PlotT
		err = cnn.gridFillInterp()
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		cnn.insertLabels()

		// At the end of all epochs, insert form previous control items in PlotT
		cnn.plot.LearningRate = strconv.FormatFloat(cnn.learningRate, 'f', 5, 64)
		cnn.plot.Epochs = strconv.Itoa(cnn.epochs)

		// Save Filters to csv file, one  per line
		f, err := os.Create(path.Join(dataDir, fileweights))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(fileweights), err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", path.Join(fileweights), err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer f.Close()

		// Save epochs, learning rate, FFT size and FFT window type
		fmt.Fprintf(f, "%d,%f,%d,%s\n", cnn.epochs, cnn.learningRate, cnn.fftSize, cnn.fftWindow)

		// Save the weights in cnn.link
		// Save Filters, 10x10 kernel and bias weights, one filter per line
		// i is the hidden layer, n is the number of Feature Maps in this layer
		for i, n := range numFMs[1:] {
			// loop over the Feature Maps in this layer
			for j := 0; j < n; j++ {
				for row := 0; row < kernelDim; row++ {
					for col := 0; col < kernelDim; col++ {
						fmt.Fprintf(f, "%.10f,", cnn.link[i][j].wgt[row][col])
					}
				}
				// last one is the bias weight and newline
				fmt.Fprintf(f, "%.10f\n", cnn.link[i][j].biaswgt)
			}
		}

		// save flattened layer, one weight per line because too long to split
		for _, wt := range cnn.wgtOutput {
			fmt.Fprintf(f, "%.10f\n", wt)
		}

		cnn.plot.Status = "MSE plotted"

		// Execute data on HTML template
		if err = tmplTrainingCNN.Execute(w, cnn.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Epochs, Learning Rate, FFT Size, and Window."
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	}
}

// Classify test examples and display test results
func (cnn *CNN) runClassification() error {
	// Loop over the training examples
	cnn.plot.Grid = make([]string, rows*cols)
	cnn.statistics =
		Stats{correct: make([]int, classes), classCount: make([]int, classes)}
	for i, samp := range cnn.samples {
		// Forward Propagation
		err := cnn.propagateForward(samp, 1)
		if err != nil {
			return fmt.Errorf("forward propagation error: %s", err.Error())
		}
		// At output layer, classify example, increment class count, %correct
		// Convert node output y to class
		err = cnn.determineClass(i, samp)
		if err != nil {
			return fmt.Errorf("determineClass error: %s", err.Error())
		}
	}

	cnn.plot.TestResults = make([]Results, classes)

	totalCount := 0
	totalCorrect := 0
	// tabulate TestResults by converting numbers to string in Results
	for i := range cnn.plot.TestResults {
		totalCount += cnn.statistics.classCount[i]
		totalCorrect += cnn.statistics.correct[i]
		cnn.plot.TestResults[i] = Results{
			Class:   strconv.Itoa(i),
			Image:   cnn.samples[i].name,
			Count:   strconv.Itoa(cnn.statistics.classCount[i]),
			Correct: strconv.Itoa(cnn.statistics.correct[i] * 100 / cnn.statistics.classCount[i]),
		}
	}
	cnn.plot.TotalCount = strconv.Itoa(totalCount)
	cnn.plot.TotalCorrect = strconv.Itoa(totalCorrect * 100 / totalCount)
	cnn.plot.LearningRate = strconv.FormatFloat(cnn.learningRate, 'f', 5, 64)
	cnn.plot.Epochs = strconv.Itoa(cnn.epochs)
	cnn.plot.FFTSize = strconv.Itoa(cnn.fftSize)
	cnn.plot.FFTWindow = cnn.fftWindow

	cnn.plot.Status = "Testing results completed."

	return nil
}

// newTestingCNN constructs an CNN from the saved cnn weights and parameters
func newTestingCNN(r *http.Request, plot *PlotT) (*CNN, error) {
	// Read in weights from csv file, ordered by layers and Feature Maps
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	// construct the CNN
	cnn := CNN{
		plot:    plot,
		samples: make([]Sample, classes),
	}
	// construct container for images
	for i := range cnn.samples {
		cnn.samples[i].image = make([][]int8, imgHeight)
		for j := range cnn.samples[i].image {
			cnn.samples[i].image[j] = make([]int8, imgWidth)
		}
	}

	// *********** Links ******************************************************

	// construct links that hold the filters
	cnn.link = make([][]Filter, hiddenLayers)
	// hidden layer filters, excluding the output layer
	// previous layer FM depth
	m := 1
	for i, n := range numFMs[1:] {
		// fully-connected hidden layers
		k := m * n
		cnn.link[i] = make([]Filter, k)
		for j := 0; j < k; j++ {
			cnn.link[i][j] = Filter{n: j, layer: i}
		}
		m = n
	}

	// outer layer uses fully connected MLP using flattened last
	// hidden layer Feature Maps = 30*15*15 = 2250, using downsampled FM
	// add bias weight with constant input = 1
	i := len(numFMs) - 1
	m = numFMs[i]*sizeFMs[i]*sizeFMs[i] + 1
	olnodes := int(math.Ceil(math.Log2(float64(classes))))
	N := m * olnodes

	// output layer links
	cnn.wgtOutput = make([]float64, N)

	// ******************* Feature Maps and Nodes ****************************
	// Input and hidden layer Feature Maps, flattened output layer nodes
	cnn.fm = make([][]FeatureMap, len(numFMs))
	cnn.node = make([][]Node, 2)

	//  input and hidden layers
	for i := 0; i < len(numFMs); i++ {
		cnn.fm[i] = make([]FeatureMap, numFMs[i])
		for j := 0; j < numFMs[i]; j++ {
			nodes := make([][]Node, sizeFMs[i])
			for k := 0; k < sizeFMs[i]; k++ {
				nodes[k] = make([]Node, sizeFMs[i])
			}
			cnn.fm[i][j] = FeatureMap{h: sizeFMs[i], w: sizeFMs[i], data: nodes}
		}
	}

	// next to last layer is the flattended last hidden layer
	cnn.node[0] = make([]Node, m)
	// init bias node to 1
	cnn.node[0][0].y = 1.0

	// output layer, which has no bias node
	cnn.node[1] = make([]Node, olnodes)

	// *************************************************************

	scanner := bufio.NewScanner(f)

	// Read in epochs, learning rate, fft size and fft window
	scanner.Scan()
	line := scanner.Text()
	items := strings.Split(line, ",")

	if len(items) != 4 {
		fmt.Printf("Require 4 parameters in %s, have %d.\n", fileweights, len(items))
		return nil, fmt.Errorf("require 4 parameters in %s, have %d", fileweights, len(items))
	}

	epochs, err := strconv.Atoi(items[0])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[0], err)
		return nil, err
	}
	cnn.epochs = epochs

	learningRate, err := strconv.ParseFloat(items[1], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[1], err)
		return nil, err
	}
	cnn.learningRate = learningRate

	fftSize, err := strconv.Atoi(items[2])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[2], err)
		return nil, err
	}
	cnn.fftSize = fftSize

	cnn.fftWindow = items[3]

	// retrieve the weights and insert in cnn.link
	// Read Filters, 10x10 kernel and bias weights
	// i is the hidden layer, n is the number of Feature Maps in this layer
loop:
	for i, n := range numFMs[1:] {
		// loop over the Feature Maps in this layer
		for j := 0; j < n; j++ {
			ok := scanner.Scan()
			if !ok {
				break loop
			}
			line := scanner.Text()
			weights := strings.Split(line, ",")
			cnn.link[i][j] = Filter{n: j, layer: i}
			k := 0
			for row := 0; row < kernelDim; row++ {
				for col := 0; col < kernelDim; col++ {
					wt, err := strconv.ParseFloat(weights[k], 64)
					if err != nil {
						fmt.Printf("ParseFloat of %s error: %v", weights[k], err)
						k++
						continue
					}
					cnn.link[i][j].wgt[row][col] = wt
					k++
				}
			}
			// last one is the bias weight
			wt, err := strconv.ParseFloat(weights[k], 64)
			if err != nil {
				fmt.Printf("ParseFloat of %s error: %v", weights[k], err)
				k++
				continue
			}
			cnn.link[i][j].biaswgt = wt
		}
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	// last layer, one weight per line
	for i := 0; i < N; i++ {
		ok := scanner.Scan()
		if !ok {
			break
		}
		line := scanner.Text()
		wgt, err := strconv.ParseFloat(line, 64)
		if err != nil {
			fmt.Printf("ParseFloat error: %v\n", err.Error())
			continue
		}
		cnn.wgtOutput[i] = wgt
	}

	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	// *******************************************************

	// construct desired from classes, binary representation
	cnn.desired = make([]float64, olnodes)

	return &cnn, nil
}

// handleTesting performs pattern classification of the test data
func handleTestingCNN(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		cnn  *CNN
		err  error
	)
	// Construct CNN instance containing CNN state
	cnn, err = newTestingCNN(r, &plot)
	if err != nil {
		fmt.Printf("newTestingCNN() error: %v\n", err)
		plot.Status = fmt.Sprintf("newTestingCNN() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Create testing examples by reading in the images
	err = cnn.createExamples()
	if err != nil {
		fmt.Printf("createExamples error: %v\n", err)
		plot.Status = fmt.Sprintf("createExamples error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// At end of all examples tabulate TestingResults
	// Convert numbers to string in Results
	err = cnn.runClassification()
	if err != nil {
		fmt.Printf("runClassification() error: %v\n", err)
		plot.Status = fmt.Sprintf("runClassification() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	filename := r.FormValue("filename")
	if len(filename) > 0 {
		// open and read the audio wav file
		// create wav decoder, audio IntBuffer, convert to audio FloatBuffer
		// loop over the FloatBuffer.Data and generate the Spectral Power Density
		// fill the grid with the PSD values
		err := cnn.processFrequencyDomain(filename)
		if err != nil {
			fmt.Printf("processFrequencyDomain error: %v\n", err)
			plot.Status = fmt.Sprintf("processFrequencyDomain error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplAudioGeneration.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		plot.Status += fmt.Sprintf("PSD of %s plotted.", filename)

	}

	// Execute data on HTML template
	if err = tmplTestingCNN.Execute(w, cnn.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// Welch's Method and Bartlett's Method variation of the Periodogram
func (cnn *CNN) calculatePSD(audio []float64, PSD []float64) (float64, float64, error) {

	N := cnn.fftSize
	m := N / 2

	// map of window functions
	window := make(map[string]Window, len(winType))
	// Put the window functions in the map
	window["Bartlett"] = bartlett
	window["Welch"] = welch
	window["Hamming"] = hamming
	window["Hanning"] = hanning
	window["Rectangle"] = rectangle

	w, ok := window[cnn.fftWindow]
	if !ok {
		fmt.Printf("Invalid FFT window type: %v\n", cnn.fftWindow)
		return 0, 0, fmt.Errorf("invalid FFT window type: %v", cnn.fftWindow)
	}
	sumWindow := 0.0
	// sum the window values for PSD normalization due to windowing
	for i := 0; i < N; i++ {
		x := cmplx.Abs(w(i, N))
		sumWindow += x * x
	}

	psdMax := -math.MaxFloat64 // maximum PSD value
	psdMin := math.MaxFloat64  // minimum PSD value

	bufm := make([]complex128, m)
	bufN := make([]complex128, N)

	// part of K*Sum(w[i]*w[i]) PSD normalizer
	normalizerPSD := sumWindow

	// Initialize the PSD to zero as it is reused when creating examples
	for i := range PSD {
		PSD[i] = 0.0
	}

	// Bartlett's method has no overlap of input data and uses the rectangle window
	// If FFT size is 8192, only use rectangle window
	if cnn.fftWindow == "Rectangle" || cnn.fftSize == 8192 {
		sections := SAMPLES / N
		start := 0
		// Loop over sections and accumulate the PSD
		for i := 0; i < sections; i++ {

			//copy(bufN, audio[i*N:])
			for j := 0; j < N; j++ {
				bufN[j] = complex(audio[start+j], 0)
			}

			// Rectangle window, unity gain

			// Perform N-point complex FFT and add squares to previous values in PSD
			// Normalize the PSD with the window sum, then convert to dB with 10*log10()
			fourierN := fft.FFT(bufN)
			x := cmplx.Abs(fourierN[0])
			PSD[0] += x * x
			for j := 1; j < N/2; j++ {
				// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
				xj := cmplx.Abs(fourierN[j])
				xNj := cmplx.Abs(fourierN[N-j])
				PSD[j] += xj*xj + xNj*xNj
			}

			// part of K*Sum(w[i]*w[i]) PSD normalizer
			normalizerPSD += sumWindow
		}

		// Normalize the PSD using K*Sum(w[i]*w[i])
		// Use log plot for wide dynamic range

		for i := range PSD {
			PSD[i] /= normalizerPSD
			PSD[i] = 10.0 * math.Log10(PSD[i])
			if i == 0 {
				fmt.Printf("PSD[0] normalized = %.3f\n", PSD[0])
			}
			if PSD[i] > psdMax {
				psdMax = PSD[i]
			}
			if PSD[i] < psdMin {
				psdMin = PSD[i]
			}
		}
		// No overlap, skip to next N samples
		start += N
		// 50% overlap sections of audio input for non-rectangle windows, Welch's method
	} else {
		// use two buffers, copy previous section to the front of current
		for j := 0; j < m; j++ {
			bufm[j] = complex(audio[j], 0)
		}
		sections := (SAMPLES - N) / m
		start := 0
		for i := 0; i < sections; i++ {
			start += m
			// copy previous section to front of current section
			copy(bufN, bufm)
			// Get the next fftSize/2 audio samples
			for j := 0; j < m; j++ {
				bufm[j] = complex(audio[start+j], 0)
			}
			// Put current section in back of previous
			copy(bufN[m:], bufm)

			// window the N samples with chosen window
			for i := 0; i < N; i++ {
				bufN[i] *= w(i, m)
			}

			// Perform N-point complex FFT and add squares to previous values in PSD
			// Normalize the PSD with the window sum, then convert to dB with 10*log10()
			fourierN := fft.FFT(bufN)
			x := cmplx.Abs(fourierN[0])
			PSD[0] += x * x
			for j := 1; j < m; j++ {
				// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
				xj := cmplx.Abs(fourierN[j])
				xNj := cmplx.Abs(fourierN[cnn.fftSize-j])
				PSD[j] += xj*xj + xNj*xNj
			}

			// part of K*Sum(w[i]*w[i]) PSD normalizer
			normalizerPSD += sumWindow
		}

		// Normalize the PSD using K*Sum(w[i]*w[i])
		// Use log plot for wide dynamic range

		for i := range PSD {
			PSD[i] /= normalizerPSD
			PSD[i] = 10.0 * math.Log10(PSD[i])
			if PSD[i] > psdMax {
				psdMax = PSD[i]
			}
			if PSD[i] < psdMin {
				psdMin = PSD[i]
			}
		}
	}

	return psdMin, psdMax, nil
}

// processFrequencyDomain calculates the Power Spectral Density (PSD) and plots it
func (cnn *CNN) processFrequencyDomain(filename string) error {
	// Use complex128 for FFT computation
	// open and read the audio wav file
	// create wav decoder, audio IntBuffer, convert IntBuffer to audio FloatBuffer
	// loop over the FloatBuffer.Data and generate the FFT
	// fill the grid with the 10log10( mag^2 ) dB, Power Spectral Density

	var (
		endpoints Endpoints
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
	)

	cnn.plot.Grid = make([]string, rows*cols)
	cnn.plot.Xlabel = make([]string, xlabels)
	cnn.plot.Ylabel = make([]string, ylabels)

	N := cnn.fftSize

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD = make([]float64, N/2)

	// Open the audio wav file
	f, err := os.Open(filepath.Join(dataDir, filename))
	if err == nil {
		defer f.Close()
		dec := wav.NewDecoder(f)
		bufInt := audio.IntBuffer{
			Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
			Data:   make([]int, SAMPLES), SourceBitDepth: bitDepth}
		_, err := dec.PCMBuffer(&bufInt)
		if err != nil {
			fmt.Printf("PCMBuffer error: %v\n", err)
			return fmt.Errorf("PCMBuffer error: %v", err.Error())
		}
		bufFlt := bufInt.AsFloatBuffer()

		// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
		psdMin, psdMax, err := cnn.calculatePSD(bufFlt.Data, PSD)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}

		endpoints.xmin = 0.0
		endpoints.xmax = float64(N / 2) // equivalent to Nyquist critical frequency
		endpoints.ymin = psdMin
		endpoints.ymax = psdMax

		// EP means endpoints
		lenEPx := endpoints.xmax - endpoints.xmin
		lenEPy := endpoints.ymax - endpoints.ymin
		prevBin := 0.0
		prevPSD := PSD[0]

		// Calculate scale factors for x and y
		xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
		yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

		// This previous cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-PSD[0])*yscale + .5)
		col := int((0.0-endpoints.xmin)*xscale + .5)
		cnn.plot.Grid[row*cols+col] = "online"

		// Store the PSD in the plot Grid
		for bin := 1; bin < N/2; bin++ {

			// This current cell location (row,col) is on the line (visible)
			row := int((endpoints.ymax-PSD[bin])*yscale + .5)
			col := int((float64(bin)-endpoints.xmin)*xscale + .5)
			cnn.plot.Grid[row*cols+col] = "online"

			// Interpolate the points between previous point and current point;
			// draw a straight line between points.
			lenEdgeBin := math.Abs((float64(bin) - prevBin))
			lenEdgePSD := math.Abs(PSD[bin] - prevPSD)
			ncellsBin := int(float64(cols) * lenEdgeBin / lenEPx) // number of points to interpolate in x-dim
			ncellsPSD := int(float64(rows) * lenEdgePSD / lenEPy) // number of points to interpolate in y-dim
			// Choose the biggest
			ncells := ncellsBin
			if ncellsPSD > ncells {
				ncells = ncellsPSD
			}

			stepBin := float64(float64(bin)-prevBin) / float64(ncells)
			stepPSD := float64(PSD[bin]-prevPSD) / float64(ncells)

			// loop to draw the points
			interpBin := prevBin
			interpPSD := prevPSD
			for i := 0; i < ncells; i++ {
				row := int((endpoints.ymax-interpPSD)*yscale + .5)
				col := int((interpBin-endpoints.xmin)*xscale + .5)
				// This cell location (row,col) is on the line (visible)
				cnn.plot.Grid[row*cols+col] = "online"
				interpBin += stepBin
				interpPSD += stepPSD
			}

			// Update the previous point with the current point
			prevBin = float64(bin)
			prevPSD = PSD[bin]

		}

		// Plot the PSD N/2 float64 values, execute the data on the plotfrequency.html template

		// Set plot status if no errors
		if len(cnn.plot.Status) == 0 {
			cnn.plot.Status = fmt.Sprintf("file %s plotted from (%.3f,%.3f) to (%.3f,%.3f)",
				filename, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
		}

	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Apply the  sampling rate in Hz to the x-axis using a scale factor
	// Convert the fft size to sampleRate/2, the Nyquist critical frequency
	sf := 0.5 * sampleRate / endpoints.xmax

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range cnn.plot.Xlabel {
		cnn.plot.Xlabel[i] = fmt.Sprintf("%.0f", x*sf)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range cnn.plot.Ylabel {
		cnn.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// handleAudioGeneration creates the audio wav files
func handleAudioGeneration(w http.ResponseWriter, r *http.Request) {

	const (
		minSines = 5 // minimum number of sinsuoids
		maxSines = 8 // maximum number of sinsuoids
	)

	var (
		plot PlotT
	)

	// get SNR from web page
	txt := r.FormValue("snr")
	filename := r.FormValue("filename")
	// Need snr or filename to continue
	if len(txt) > 0 || len(filename) > 0 {
		if len(txt) > 0 {
			snr, err := strconv.Atoi(txt)
			if err != nil {
				fmt.Printf("SNR int conversion error: %v\n", err)
				plot.Status = fmt.Sprintf("SNR conversion to int error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplAudioGeneration.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}

			// Delete the current wav files
			files, err := os.ReadDir(dataDir)
			if err != nil {
				fmt.Printf("ReadDir for %s error: %v\n", dataDir, err)
				plot.Status = fmt.Sprintf("ReadDir %s error: %v", dataDir, err)
				// Write to HTTP using template and grid
				if err := tmplAudioGeneration.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}

			for _, dirEntry := range files {
				name := dirEntry.Name()
				if filepath.Ext(name) == ".wav" {
					if err := os.Remove(path.Join(dataDir, name)); err != nil {
						plot.Status = fmt.Sprintf("Remove %s error: %v", name, err)
						// Write to HTTP using template and grid
						if err := tmplAudioGeneration.Execute(w, plot); err != nil {
							log.Fatalf("Write to HTTP output using template with error: %v\n", err)
						}
						return
					}
				}
			}

			// Calculate the noise standard deviation (SD) using the SNR and maxampl of sine
			ratio := math.Pow(10.0, float64(snr)/10.0)
			// Amplitude sine waves, make big enough to hear in the media player
			maxampl := 500.0
			noiseSD := math.Sqrt(0.5 * maxampl * maxampl / ratio)

			// define sine wave frequencies for k*120 Hz, k=1,2,...,16, gives 120 to 1920
			// 2000 Hz is the Nyquist frequency since 4000 Hz is the sampling rate
			freq := make([]float64, classes)
			const delFreq = 120.0
			for i := range freq {
				freq[i] = float64(i+1) * delFreq
			}

			// time step is 1/sampleRate
			const delT = 1.0 / sampleRate

			// loop over the classes and generate a wav file for each one
			for class := 0; class < classes; class++ {

				// randomly select the number of sines and the frequencies for this class
				nsines := rand.Intn(maxSines-minSines+1) + minSines
				fflt64 := make([]float64, nsines)
				fstr := make([]string, nsines)
				trial := 0
				// Disallow duplicate frequencies
				for i := range fflt64 {
					for {
						dupl := false
						trial = rand.Intn(classes)
						for j := range fflt64 {
							if fflt64[j] == freq[trial] {
								dupl = true
								break
							}
						}
						if !dupl {
							break
						}
					}
					fflt64[i] = freq[trial]
					fstr[i] = strconv.Itoa(int(fflt64[i]))
				}
				// compose wav file name from constituent frequencies
				wav_file := strings.Join(fstr, "_") + ".wav"

				// start time for every audio wav file
				t := 0.0

				// loop over the samples
				for samp := 0; samp < SAMPLES; samp++ {
					sum := 0.0
					// loop over the sinusoidal frequencies
					for _, f := range fflt64 {
						sum += maxampl * math.Sin(twoPi*f*t)
					}
					// add Gaussian noise with zero mean and given SD
					sum += noiseSD * rand.NormFloat64()

					// save the sample
					audioSamples[samp] = sum

					// increment the time
					t += delT
				}

				//   create wav file
				outF, err := os.Create(path.Join(dataDir, wav_file))
				if err != nil {
					fmt.Printf("os.Create() file %s error: %v\n", wav_file, err)
					plot.Status = fmt.Sprintf("os.Create() file %s error: %v", wav_file, err.Error())
					// Write to HTTP using template and grid
					if err := tmplAudioGeneration.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
				defer outF.Close()

				// create wav.Encoder
				enc := wav.NewEncoder(outF, sampleRate, bitDepth, 1, 1)

				// create audio.FloatBuffer
				float64Buf := &audio.FloatBuffer{Data: audioSamples, Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate}}

				// create IntBuffer from FloatBuffer and pass to Encoder.Write()
				if err := enc.Write(float64Buf.AsIntBuffer()); err != nil {
					fmt.Printf("wav encoder write error: %v\n", err)
					plot.Status = fmt.Sprintf("wav encoder write error: %v", err.Error())
					// Write to HTTP using template and grid
					if err := tmplAudioGeneration.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}

				// close the encoder
				if err := enc.Close(); err != nil {
					fmt.Printf("wav encoder close error: %v\n", err)
					plot.Status = fmt.Sprintf("wav encoder error: %v", err.Error())
					// Write to HTTP using template and grid
					if err := tmplAudioGeneration.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}

			}

			plot.Status = "audio wav files generated"

			// Generate the PSD of the wav file and plot
		} else if len(filename) > 0 {
			// open and read the audio wav file
			// create wav decoder, audio IntBuffer, convert to audio FloatBuffer
			// loop over the FloatBuffer.Data and generate the Spectral Power Density
			// fill the grid with the PSD values
			fftWindow := r.FormValue("fftwindow")
			txt = r.FormValue("fftsize")
			if len(fftWindow) == 0 || len(txt) == 0 {
				fmt.Printf("missing FFT window %s or FFT size %s\n", fftWindow, txt)
				plot.Status = fmt.Sprintf("missing FFT window %s or FFT size %s\n", fftWindow, txt)
				// Write to HTTP using template and grid
				if err := tmplAudioGeneration.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
			}
			fftSize, err := strconv.Atoi(txt)
			if err != nil {
				fmt.Printf("fftsize int conversion error: %v\n", err)
				plot.Status = fmt.Sprintf("fftsize int conversion error: %v\n", err)
				// Write to HTTP using template and grid
				if err := tmplAudioGeneration.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
			}

			cnn := CNN{plot: &plot, fftSize: fftSize, fftWindow: fftWindow}
			err = cnn.processFrequencyDomain(filename)
			if err != nil {
				fmt.Printf("processFrequencyDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processFrequencyDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplAudioGeneration.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}
		// Execute data on HTML template
		if err := tmplAudioGeneration.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	} else {
		plot.Status = "Enter either SNR or an audio wav filename.  Choose FFT Size and Window."
		// Write to HTTP using template and grid
		if err := tmplAudioGeneration.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	}
}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for training and testing the CNN Neural Network

	// Create HTTP handler for training
	http.HandleFunc(patternTrainingCNN, handleTrainingCNN)
	// Create HTTP handler for testing
	http.HandleFunc(patternTestingCNN, handleTestingCNN)
	// Create HTTP handler for generating the wav audio files
	http.HandleFunc(patternAudioGeneration, handleAudioGeneration)
	fmt.Printf("Convolutional Neural Network Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
