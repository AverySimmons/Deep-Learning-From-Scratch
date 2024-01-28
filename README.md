
# Deep Learning From Scratch

---

**Tools Used:** C &nbsp;&nbsp;&nbsp;&nbsp; **Keywords:** Gradient decent, Memory managment, File parsing, Test driven development

---

### Description:
&nbsp;&nbsp;&nbsp;&nbsp;This project aims to reinvent the deep learning wheel entirely from scratch in C, with comprehensive test coverage. I have always been fascinated by the inner workings of deep learning. In order to gain a deeper understanding of the so-called black box, and to greaten my technical knowledge of low-level programming, I have decided to create a codebase from scratch. This codebase is designed to train a deep neural network on data from a CSV file using gradient descent.

### Features:
- &nbsp;&nbsp;&nbsp;&nbsp;**Low Level:**  
Since this project is build in C all memory allocation and garbage collection is done manually and all data structures are custom made. For example each neural network is a structure composed of layers that are each two allocated array's, one of weights and the other of biases. Each of these structures is allocated and freed using functions that accept the size of the network's layers as inputs.

- &nbsp;&nbsp;&nbsp;&nbsp;**Deep Learning:**  
Network's in this project are densly connected with ReLU activation functions and use gradient decent through back propogation to learn. Both forward and backward passes through the network are acomplished through matrix multiplication, an interesting feature here is that when training the networks feed forward and backward subsets of the data simultaneously leading to some three dimensional matrices.  
(an addition note is that really the matrix multiplication should be parallelized across the gpu, originally I was using the CBLAS software to acomplish this, but ran into too many problems to make it worth the effort.)

- &nbsp;&nbsp;&nbsp;&nbsp;**Test Driven:**  
All functions in this project have full test coverage. At first I thought this might be too much work, but after realizing how hard it would be to find mathmatical errors within individual functions (because these errors would only result in the model learning slower or not at all), I decided that tests were a necessity.

- &nbsp;&nbsp;&nbsp;&nbsp;**Input File Parsing:**  
Because this project is quite low level I had to manually create a custom file parsing system for csv files. This data is read into a data structure that can be used as an input for training functions.


### Code Breakdown:
- &nbsp;&nbsp;&nbsp;&nbsp;**Matrix Operations:**  
General matrix operations such as addition, multiplication, etc. as well as ReLU and ReLU prime functions.

- &nbsp;&nbsp;&nbsp;&nbsp;**Neural Network:**  
Data structures and functions for allocating, freeing, and training neural networks.

- &nbsp;&nbsp;&nbsp;&nbsp;**Data Loading:**  
Logic needed to load csv file data into a input data structure, which can be used for functions inisde the Neural Network file.
