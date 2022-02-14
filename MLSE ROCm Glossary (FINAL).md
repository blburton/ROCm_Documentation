

# MLSE ROCm Software Glossary
---
# [Notes](#Notes-ID)


This glossary is to track definitions and terms as they pertain to the ROCm documentation, any additions made should be added alphabetically. This is the master list, to update please submit your terms and definitions to ROCmDocsSupport@amd.com.

---


# [A](#A-ID)
|Term  |Definition  |
| --- | --- |
| _**accelerator**_ | Increases the speed a program or functions operates. |
| _**active host thread**_ | The thread which is running HIP APIs. |
| _**API**_|  Application Programming Interface, a type of software interface that allows the connectivity between computers and/or programs.  |
| _**Apiary**_ |  A platform for API design that allows developers to prototype APIs. |
|  _**artificial intelligence**_ |  Machines/Computer lead intelligence. |
| _**AQL Architecture Queueing Language**_ |   The Architected Queuing Language (AQL) is a standard binary interface used to describe commands such as a kernel dispatch. An AQL packet is a user-mode buffer with a specific format that encodes one command. AQL allows agents to build and enqueue their own command packets, enabling fast, low-power dispatch. AQL also provides support for kernel agent queue submissions: the kernel agent kernel can write commands in AQL format. |

---
# [B](#B-ID)
|Term  |Definition  |
| --- | --- |
| _**bandwidth**_ | The max amount of data that can be transferred. |
|  _**BLAS**_ |  Basic Linear Algebra Subprogram, routines for performing linear algebra operations. |
|  _**byte**_ |  Unit of digital data. |

---
# [C](#C-ID)
|Term  |Definition  |
| --- | --- |
| _**C++ complier**_ |  Software that translates C++ source code. |
|  _**categorical data**_ |  Data type which is divided into groups/classifications. |
|  _**CentOS**_ |  A Linux distribution for an open-source community.  |
|  _**code**_ |  See: source code |
| _**completion\_future**_ | becomes ready, &quot;Completes&quot; |
| _**compute or computation**_ |   Traditionally: calculationFor technical use, compute can replace computation. Use of computer, mathematical calculations in relation to computers. |
|  _**compute units**_ |  Unit of measurement for processing resources for GPUs. |
|  _**command**_ |  An instruction to an application. |
|  _**conditionals**_ |  A sequence of program commands that evaluate to be either ture or false. |
|  _**confusion matrix**_ |  Table used for the evaluation of algorithm performance. Compares actual and predicted values by machine learning model. |
|  _**convolution**_ |  An operation that interlaces two sources of data, utilized in image processing. |
|  _**convolutional layer**_ |  A layer that applies a convolution operation to the input. |
|  _**convolutional neural network**_ |  A neural network for processing arrays of image data. |
|  _**core**_ |  Paired with adjective to describe processor type. E.G. Single-core, dual-core, etc. |
|  _**CPU (Central Processing Unit)**_ |  The processor that executes commands within a computer program. |
|  _**cross-validation**_ |  The process of testing the abilities of a machine learning model. |
| _**CUDA(R)**_ |  The CUDA environment for statistical computing and graphics. |
| _**curl**_ | Contrastive Unsupervised Representation Learning |

---
# [D](#D-ID)

|Term  |Definition  |
| --- | --- |
| _**data set (dataset)**_ | A collection of data.
| _**DataFrame**_ |  A two-dimensional table of data, like a spreadsheet. |
| _**debug**_ |  The process of locating and resolving program errors within software.  |
| _**decoder**_ |  A machine learning system that converts internal representation to external representation. |
| _**default device**_ | Each host thread maintains a default device. Most HIP runtime APIs (including memory allocation, copy commands, kernel launches) do not use accept an explicit device argument but instead implicitly use the default device. The default device can be set with hipSetDevice. |
| _**default value**_ | A standard setting within the program that assigns a value automatically if one is not entered by the user. |
|  _**dimension reduction**_ |  Reduction of the features dimensions. A technique that aids in the performance for machine learning algorithms. |
|  _**downstream**_ |  Using developed libraries to build an application. Refers to the flow of code, flows away from origin. |
| _**driver**_ | A program that applies internal functions to the OS and activates said functions. |

---
# [E](#E-ID)
|Term  |Definition  |
| --- | --- |
| _**encoder**_ | A machine learning system that converts from an external representation into an internal representation. |
| _**endpoints**_ | One end of the communication channel an API interacts with the system. |
| _**environment**_ | The conditions that support the performance of a function. |

---
# [F](#F-ID)
|Term  |Definition  |
| --- | --- |
| _**FALSE**_ | Return value, the execution of the function is prevented. |
|  _**feature extraction**_ | The transformation of raw data into numerical features while still preserving original data. |
|  _**feature vector**_ | A vector of numerical features that represents an object. The combination of feature vectors makes up feature space. |
|  _**feedback loop**_ |  When the system&#39;s output is utilized as inputs to improve performance. |
|  _**feedforward neural network (FFN)**_ |  A neural network without recursive connections. |
|  _**fine tuning**_ |  A way to apply and utilize transfer learning by changing the model output to fit a new task. |
|  _**Frameset**_ |  A part of HTML that contains different frame elements. |
|  _**function (method)**_ |  The action of the API. |

---
# [G](#G-ID)
|Term  |Definition  |
| --- | --- |
| _**GAN**_ | Generative Adversarial Networks, a generative model that creates new data instance that resemble training data. |
|  _**Gigabyte**_ |  Unit symbol GB. Gigabyte or gig is a unit of byte for computer content. One billion bytes. |
|  _**GitHub**_ |  An open-source community for code hosting and collaboration. |
|  _**GPU (Graphics Processing Unit)**_ |  A specialized processor dedicated for output graphics display.  |
|  _**GRUB**_ |  A multiboot boot loader. |

---
# [H](#H-ID)
|Term  |Definition  |
| --- | --- |
| _**hardware**_ | Physical machines |
| _**HCC (Heterogeneous Compute Compiler) (Deprecated)**_ |  An Open Source, Optimizing C++ Compiler for Heterogeneous Compute. It supports heterogeneous offload to AMD APUs and discrete GPUs via HSA enabled runtimes and drivers. It is based on Clang, the LLVM Compiler Infrastructure and the &#39;libc++&#39; C++ standard library. The goal is to implement a compiler that takes a program that conforms to a parallel programming standard such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP, and transforms it into the AMD GCN ISA. |
|  _**hidden layer**_ |  The layer between input layers and output layers. Artificial neurons intake inputs and produce an output. |
| _**HIP (Heterogeneous Interface for Portability)**_ |  Heterogeneous Interface for Portability is a C++ runtime API and kernel language that allows developers to create portable applications that can run on AMD and other GPU&#39;s. It provides a C-style API and a C++ kernel language. The first big feature available in the HIP is porting apps that use the CUDA Driver API. |
|  _**HIP-Clang**_ |  Heterogeneous AMDGPU Compiler, with its capability to compile HIP programs on AMD platform. |
|  _**hipconfig**_ |  Tool to report various configuration properties of the target platform. |
|  _**hipify tools**_ |  Tools to convert CUDA® code to portable C++ code. |
|  _**hipSetDevice**_ |  A set default device for hip API calls. |
|  _**host, host CPU**_ |  Executes the HIP runtime API and is capable of initiating kernel launches to one or more devices. |
|  _**HPC (High  Performance Computing)**_ |  The ability to perform and process data at high speeds. |
| _**HSA (Heterogeneous System Architecture)**_ | HSA provides a unified view of fundamental computing elements. HSA allows a programmer to write applications that seamlessly integrate CPUs (called latency compute units) with GPUs (called throughput compute units), while benefiting from the best attributes of each. HSA creates an improved processor design that exposes the benefits and capabilities of mainstream programmable compute elements, working together seamlessly. HSA is all about delivering new, improved user experiences through advances in computing architectures that deliver improvements across all four key vectors: improved power efficiency; improved performance; improved programmability; and broad portability across computing devices. For more on [HSA](http://developer.amd.com/wordpress/media/2012/10/hsa10.pdf). |
| _**HTC (High Throughput Computing)**_ | Allows system to run multiple software across multiple processors and the same time.  Refers to the computing over long periods of time. |
| _**hyperarameter**_ | Values that control learning process using learning algorithm. |

---
# [I](#I-ID)
|Term  |Definition  |
| --- | --- |
| _**image recognition**_ | The systems or software ability to identify objects or details in an image. |
|  _**imbalanced dataset**_ |  Datasets where the target class is unequally distributed. |
|  _**inference**_ |  Running live data points into a machine learning algorithm. |

---
# [J](#J-ID)
|Term  |Definition  |
| --- | --- |
| _**Java, JavaScript, JScript**_ | An object-oriented programming language. |
|  _**Jira**_ |  A bug tracking and agile project management software. |

---
# [K](#K-ID)
|Term  |Definition  |
| --- | --- |
| _**Keras**_ | An open-source library that acts as an interface for TensorFlow library. |
|  _**Kernel**_ |  Program working within the operating system that controls the system. Facilitates interactions between hardware and software.  |
| _**Kernel GPU Driver**_ |  Software program that runs within the OS that between the applications, and graphics card components. |
| _**KVM**_ |  Kernel-based Virtual Machine, an open-source virtualization within Linux that allows a host of machine to run multiple virtual environments. |

---
# [L](#L-ID)
|Term  |Definition  |
| --- | --- |
| _**learning rate**_ | The hyperparameter that controls the rate/speed that a model learns. |
| _**Linux**_ | An open-source OS, that manages the system&#39;s hardware, CPU, memory, and storage. |
| _**Long Short-Term Memory (LSTM)**_ | A recurrent neural network that processes data long-term. |

---
# [M](#M-ID)
|Term  |Definition  |
| --- | --- |
| _**machine learning**_ | The use and development of computer systems with the capability of learning and adapting using algorithms. |
| _**matplotlib**_ | An open-source library within Python that visualizes data. |
| _**memory**_ | Refer to specific type. RAM, ROM, hard drive, local, etc. |
|  _**method (function)**_ |  An action performed by the API. The method resides within the API. |
|  _**micro-architecture**_ |  The specific design of a microprocessor. |
| _**MIOpen**_ |  AMD&#39;s deep learning primitive&#39;s library which provides highly optimized, and hand-tuned implementations of different operators such as convolution, batch normalization, pooling, softmax, activation and layers for Recurrent Neural Networks (RNNs), used in both training and inference. |
| _**MLPerf**_ |  Utilized for measuring the speed in which systems can run models in different scenarios.  |
| _**MNIST**_ |  A large database of handwritten digits used for training image processing systems. |
| _**model training**_ |  The phase in development where datasets are used to train the Machine Learning algorithm. |

---
# [N](#N-ID)
|Term  |Definition  |
| --- | --- |
| _**neural network**_ | A network of functions used to translate data input into a desire output. |
| _**neuron**_ | A function that takes a group of inputs and process them to be the output for the next layer. Its purpose is to model the actual functioning of a biological neuron. |
| _**node (neural network)**_ | Neuron in a hidden layer. |
| _**node (TensorFlow graph)**_ | An operation in TensorFlow graph. |
| _**nodes**_ | A unit that has one or more weighted input connections. See: neuron |
| _**normalization**_ | The reorganization of data into the same scale. |
| _**null**_ | Null value |
| _**NumPy**_ | A linear algebra library in Python, it supports mutli-dimensional arrays and matrices. |
| _**nvcc**_ | nvcc compiler, do not capitalize. |

---
# [O](#O-ID)
|Term  |Definition  |
| --- | --- |
| _**offline inference**_ |The process of generating predictions on batch of observations that occur on a recurring schedule. |
|  _**one-hot encoding**_ |  The method of converting data into a form for Machine Learning algorithms. |
|  _**OpenACC**_ |  For accelerators, a programming standard for parallel computing on accelerators. |
| _**OpenCL**_ |  Open Computing Language (OpenCL) is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators. OpenCL provides a standard interface for parallel computing using task- and data-based parallelism. The programming language that is used to write compute kernels is called OpenCL C and is based on C99,[16] but adapted to fit the device model in OpenCL. OpenCL consists of a set of headers and a shared object that is loaded at runtime. As of 2016 OpenCL runs on Graphics processing units, CPUs with SIMD instructions, FPGAs, Movidius Myriad 2, Adapteva epiphany and DSPs. |
|  _**OpenMP**_ |  Programming model, supports offloading code and data onto GPU devices. |
|  _**Open MPI**_ |  A Message Passing Interface (MPI) library project. |
|  _**open source**_ | Software for which the source code is publicly accessible. |
|  _**optimizer**_ | A program that utilizes linear programming to optimize processes. |
|  _**outliers**_ | Values that are abnormal. |
|  _**overfitting**_ |  A model learns data to the extent that it negatively impact performance of the model on new data. |
|  _**oversampling**_ |  Duplicate examples from minority class. Data analysis technique that adjusts the class distribution of a dataset. |

---
# [P](#P-ID)
|Term  |Definition  |
| --- | --- |
| _**Pandas**_ | An open-source data analysis tool built on top of Python programing language. |
|  _**Parameters**_ |  Displayed in parentheses when written in a function, parameters act as variables that assign value. |
| _**PCIe Platform Atomics**_ | PCI Express (PCIe) was developed as the next generation I/O system interconnect after PCI, designed to enable advanced performance and features in connected devices while remaining compatible with the PCI software environment. Today, atomic transactions are supported for synchronization without using an interrupt mechanism. In emerging applications where math co-processing, visualization and content processing are required, enhanced synchronization would enable higher performance. |
| _**perceptron**_ |  A mathematical model of a biological neuron. |
|  _**PerfZero**_ |  Framework for TensorFlow.  &quot;...consolidating the docker image build, GPU driver installation, TensorFlow installation, benchmark library checkout, data download, system statistics collection, benchmark metrics collection, profiler data collection and so on into 2 to 3 commands. &quot; |
|  _**precision**_ |  The quality of positive predictions made. |
| _**precision-recall curve**_ |  The performance metrics between precision and recall for binary classification algorithms. |
| _**prediction**_ |  The output of an algorithm on a set of data based on predictive analytics. |
|  _**prediction bias**_ |  The difference between a model apparent and prediction error. |
| _**pre-training model**_ |  A model created based on already created and trained models. |
| _**probabilistic regression model**_ |  Predictions made on a dependent variable based on information |
| _**PyTorch**_ |  An open-source machine learning library used for Python. |
| _**pre-training model**_ |  A model created based on already created and trained models. |

---
# [Q](#Q-ID)
|Term |Definition  |
| --- | --- |
| _**Queue**_ | A Queue is a runtime-allocated resource that contains a packet buffer and is associated with a packet processor. The packet processor tracks which packets in the buffer have already been processed. When it has been informed by the application that a new packet has been enqueued, the packet processor is able to process it because the packet format is standard, and the packet contents are self-contained -- they include all the necessary information to run a command. A queue has an associated set of high-level operations defined in &quot;HSA Runtime Specification&quot; (API functions in host code) and &quot;HSA Programmer Reference Manual Specification&quot; (kernel code). |

---
# [R](#R-ID)
|Term  |Definition  |
| --- | --- |
| _**rank (Tensor)**_ | A unit of dimensionality, the number of dimensions of the tensor within the TensorFlow system. |
|  _**RCCL**_ |  A stand-alone library of standard collective communication routines for GPUs. |
|  _**recall**_ |  The true positives found. The capability for a model to find relevant cases in a dataset. |
|  _**recommendation system**_ |  A system designed for predicting user responses. |
|  _**Rectified Linear Unit (ReLU)**_ |  A non-linear activation function that will output the input directly if it is positive. |
|  _**recurrent neural network**_ |  An artificial neural network set to work for time series data. |
|  _**regression model**_ |  A statistical analysis for estimate the connection between outcome variable and predictors. |
|  _**regularization**_ | A form of regression reduces error by forming a function on the training set. |
|  _**regularization rate**_ | A scalar value that specifies the relative importance of regularization function. Represented as lambda. |
| _**representation**_ | The process of mapping data to useful features. |
|  _**regression model**_ |  A statistical analysis for estimate the connection between outcome variable and predictors.|
|  _**regularization**_ | A form of regression reduces error by forming a function on the training set. |
|  _**regularization rate**_ |  A scalar value that specifies the relative importance of regularization function. Represented as lambda. |
| _**representation**_ | The process of mapping data to useful features. |
|  _**REST API**_ | A style of networked applications limited to a client-server-based applications.  |
|  _**RBT**_ | Risk-Based Testing |
| _**RHEL**_ | A subscription-based operating system. You must enable the external repositories to install on the devtoolset-7 environment and the dkms support files. |
| _**RNG, FFT**_ | The random number generator and fast fourier transform algorithms have many applications in creating GPU applications for different fields. |
| _**ROCclr**_ | A virtual device interface that compute runtimes interact with different backends such as ROCr on Linux or PAL on Windows. The [ROCclr](https://github.com/ROCm-Developer-Tools/ROCclr) is an abstract layer allowing runtimes to work on both OS without much effort. |
| _**ROCdbgapi**_ | The AMD Debugger API is a library that provides all the support necessary for a debugger and other tools to perform low level control of the execution and inspection of execution state of AMD&#39;s commercially available GPU architectures. |
|  _**rocFFT**_ | An implementation of Fast Fourier Transform (FFT) written in HIP for GPUs. |
| _**rocGDB**_ | ROCm source-level debugger for Linux. |
| _**ROC (receiver operating characteristic) Curve**_ | A curve of true positive rate vs. false positive rate at different classification thresholds. |
|  _**ROCm SMI**_ |ROCm System Management Interface, command line interface for manipulating and monitoring the amdgpu kernel.|
| _**ROCm Validation Suite**_ |The ROCm Validation Suite (RVS) is a system administrator&#39;s and cluster manager&#39;s tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform. |
| _**rocProfiler**_ |A command line tool used for API environment configurations. For profiling GPU compute applications. |
|  _**rocSOLVER**_ |  An implementation of LAPACK routines on top of the AMD&#39;s open source ROCm platform. rocSOLVER is implemented in the HIP programming language and optimized for AMD&#39;s latest discrete GPUs. |
| _**rocSPARSE**_ |ROCm library that contains basic linear algebra subroutines for sparse matrices and vectors written in HiP for GPU devices. It is designed to be used from C and C++ code.|
| _**ROCr ROCm runtime**_ |  The HSA runtime is a thin, user-mode API that exposes the necessary interfaces to access and interact with graphics hardware driven by the AMDGPU driver set and the ROCK kernel driver. Together they enable programmers to directly harness the power of AMD discrete graphics devices by allowing host applications to launch compute kernels directly to the graphics hardware . **Accelerator Modes Supported:** HC C++ API, HIP, C++ AMP, C++ Parallel STL, OpenMP |
| _**ROCT-THUNK/ROC Runtime**_ | A cmake-based system available for building thunk. |
| _**rocThrust**_ | A parallel algorithm library, runs Thrust dependent software on AMD GPUs. |
| _**rocTracer**_ | Runtime API calls and asynchronous GPU activity tracer. |
| _**Runtime**_ |  The execution of a program. |

---
# [S](#S-ID)
|Term  |Definition  |
| --- | --- |
| _**sampling bias**_ | Collected data does not accurately reflect the program expected to run. |
|  _**scalar**_ | Consisting of a single value (e.g. integer or string) rather than multiple values (e.g. array). |
|  _**scaling**_ |  The expansion of configuration to handle/manage the load amount on the server. |
|  _**scikit-learn**_ |  Python&#39;s machine learning library, features clustering algorithms, regression, and classification. |
| _**scoring**_ |  The action of building an algorithmic model from historical dataset to a new dataset. |
| _**script**_ |  Embedded code. |
|  _**SDK**_ |  Software development kit, used to implement API.  |
|  _**selection bias**_ |  When a dataset is not reflective of reality of the environment the model runs. |
|  _**self-supervised learning**_ |  A machine learning technique where the model learns by using unlabeled data to predict and generate missing information. |
|  _**self-training**_ |  See: semi-supervised learning |
| _**semi-supervised learning**_ |  An approach that trains using a small amount of label data and the model makes predictions of unlabeled data. A combination of supervised and unsupervised training. |
| _**Shader Engines**_ | A program that tells the model how to render pixels. |
| _**shape (tensor)**_ | The number of elements in dimensions. |
| _**source code**_| Code written by developers for a computer program.
| _**Sparse**_ | Some problems are defined in terms of linear operations over arrays of data (e.g. vectors and matrices) the elements of which are mostly zeros (sparse arrays). When the fraction of zeros is significantly large, enough so that there are benefits to explicitly take these zeros into account when solving the problem, these problems are called Sparse Linear Algebra problems. |
|  _**squared loss**_ | A function that is used to indicate how much predicted output is off. |
|  _**staged training**_ | A training method for a model that forms in stages. |
|  _**supervised machine learning**_ |  Training model that utilizes input data and corresponding labels. |

---
# [T](#T-ID)
|Term  |Definition  |
| --- | --- |
| _**tensor processing unit**_ | Google&#39;s custom application-specific integrated circuits that is used to accelerate machine learning workloads. |
|  _**tensor shape**_ |  Number of elements a Tensor contains in various dimension. |
|  _**tensor size**_ |  Number of scalars a Tensor contains. |
| _**TensorFlow**_ | An end-to-end open-source platform used for machine learning, contains a system of tools, libraries and community resources that allows developers to build machine learning powered applications |
|  _**tensorflow playground**_ | An interactive visualization of neural networks. |
| _**time series**_ |The sequence of observations collected in constant time intervals.|
|  _**token**_ | An instance of a sequence of characters in some document that grouped together as a useful semantic unit for processing. |
|  _**training**_ | The process of learning a machine learning model using algorithms and training data. |
|  _**training set**_ |  A set of data examples provided during the learning process.|
| _**transformer**_ | A model architecture for transforming one sequence into another using Encoder and Decoder. |
| _**True negative**_ | An output where the model predicts a negative class correctly. |
| _**True positive**_ | An output where the model predicts a positive class correctly. |

---
# [U](#U-ID)
|Term  |Definition  |
| --- | --- |
| _**Ubuntu**_ | An open-source Linux distribution. |
| _**unsupervised machine learning**_ |  Utilizing algorithms to analyze unlabeled dataset. The algorithm detects hidden patterns and data groupings without a human user. |
| _**upstream**_ |Reference to the flow of open-source code. If code developed from an origin however, it is incorporated into original source code. |

---
# [V](#V-ID)
|Term  |Definition  |
| --- | --- |
| _**validation**_ | The use of a testing data set to evaluate a training model. |
| _**vanishing gradient**_ |  When useful gradient information is unable to be propagated from the output end of the model back to the layers near the input end of the model. |
|  _**variable**_ |  Assigned within the function and carries data input throughout the function. |

---
# [W](#W-ID)

---
# [X](#X-ID)

---
# [Y](#Y-ID)

---
# [Z](#Z-ID)
## [Reference Sources](#referencesources-ID)
---

|     Sources                     |                         |
|--------|-----------------------------|
|   **Microsoft** |[Reference Source](https://docs.microsoft.com/en-us/security-updates/glossary/glossary)      |
|**RedHat**| [Reference Source](https://access.redhat.com/documentation/en-us/red_hat_satellite/6.2/html/architecture_guide/appe-red_hat_satellite-architecture_guide-glossary_of_terms)|
|**Google**|[Reference Source](https://developers.google.com/machine-learning/glossary)  |  
|**PyTorch**|[Reference Source](https://pytorch.org/docs/stable/community/contribution_guide.html)|
|**Intel**|[Reference Source](https://www.intel.com/content/www/us/en/support/topics/glossary.html) |
|**GSA's Digital.gov**|[Reference Source](https://digital.gov/2015/08/03/18fs-style-guide-for-open-source-project-documentation/)  |
|**AMD Developer Central**| [Reference Source](https://developer.amd.com/resources/rocm-learning-center/)
