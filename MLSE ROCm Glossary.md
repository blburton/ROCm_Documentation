# MLSE ROCm Software Glossary
---



|Term                          |Definition                         |
|--------|-----------------------------|
|   **_accelerator_** |Increases the speed a program or functions operates. |
|**_active host thread_**|The thread which is running HIP APIs.
|**_API_**|Application Programming Interface, a type of software interface that allows the connectivity between computers and/or programs.
|**_Apiary_**|A platform for API design that allows developers to prototype APIs.|
|**_artificial intelligence_**|Machines/Computer lead intelligence.|
|**_AQL_**|The Architected Queuing Language (AQL) is a standard binary interface used to describe commands such as a kernel dispatch. An AQL packet is a user-mode buffer with a specific format that encodes one command. AQL allows agents to build and enqueue their own command packets, enabling fast, low-power dispatch. AQL also provides support for kernel agent queue submissions: the kernel agent kernel can write commands in AQL format.|
|**_bandwidth_**|The max amount of data that can be transferred.|
|**_BLAS_**|Basic Linear Algebra Subprogram, routines for performing linear algebra operations.
|**_byte_**|Unit of digital data.
|**_C++ complier_**|Software that translates C++ source code.|
|**categorical data**|Data type which is divided into groups/classifications.|
|**_CentOS_**|A Linux distribution for an open-source community.|
|**_code_**|See: source code|
|**_completion_future_**|becomes ready, "Completes"
|**_compute or computation_**|Traditionally: calculation  For technical use, compute can replace computation. Use of computer, mathematical calculations in relation to computers.|
|**_Compute Units_**|Unit of measurement for processing resources for GPUs.|
|**_command_**|An instruction to an application.|
|**_conditionals_**|A sequence of program commands that evaluate to be either true or false. 
|**_confusion matrix_**|Table used for the evaluation of algorithm performance. Compares actual and predicted values by machine learning model.|
|**_convolution_**|An operation that interlaces two sources of data, utilized in image processing.|
|**_convolutional layer_**| A layer that applies a convolution operation to the input.|
|**_convolutional neural network_**|A neural network for processing arrays of image data.|
|**_core_**|Paired with adjective to describe processor type. EG. single-core, dual-core, etc.|
|**_CPU (Central Processing Unit)_**|The processor that executes commands within a computer program.|
|**_cross-validation_**|The process of testing the abilities of a machine learning model.|
|**_CUDA(R)_**|The CUDA environment for statistical computing and graphics.|
|**_curl_**|Contrastive Unsupervised Representation Learning|
|**_data set (dataset)_**|A collection of data.|
|**_DataFrame_**|A two-dimensional table of data, like a spreadsheet.|
|**_debug_**|The process of locating and resolving program errors within software.|
|**_decoder_**|A machine learning system that converts internal representation to external representation.
|**_default device_**|Each host thread maintains a default device. Most HIP runtime APIs (including memory allocation, copy commands, kernel launches) do not use accept an explicit device argument but instead implicitly use the default device. The default device can be set with hipSetDevice.|
|**_default value_**|A standard setting within the program that assigns a value automatically if one is not entered by the user.|
|**_dimension reduction_**|Reduction of the features dimensions. A technique that aids in the performance for machine learning algorithms.|
|**_downstream_**|Using developed libraries to build an application. Refers to the flow of code, flows away from origin.
|**_driver_**|A program that applies internal functions to the OS and activates said functions.
|**_encoder_**|A machine learning system that converts from an external representation into an internal representation
|**_endpoints_**|One end of the communication channel an API interacts with the system.
|**_environment_**|The conditions that support the performance of a function.
|**_FALSE_**|Return value, the execution of the function is prevented.
|**_feature extraction_**|The transformation of raw data into numerical features while still preserving original data.
|**_feature vector_**|A vector of numerical features that represents an object. The combination of feature vectors makes up feature space.
|**_feedback loop_**|When the system's output is utilized as inputs to improve performance.
|**_feedforward neural network (FFN)_**|A neural network without recursive connections.
|**_fine tuning_**|A way to apply and utilize transfer learning by changing the model output to fit a new task.
|**_Frameset_**|A part of HTML that contains different frame elements.
|**_function (method)_**|The action of the API.
|**_GAN_**|Generative Adversarial Networks, a generative model that creates new data instances that resemble training data.
|**_Gigabyte_**|Unit symbol GB. Gigabyte or gig is a unit of byte for computer content. One billion bytes.
|**_GitHub_**|An open-source community for code hosting and collaboration.
|**_GPU (Graphics Processing Unit)_**|A specialized processor dedicated for output graphics display.
|**_GRUB_**|A multiboot boot loader.
|**_hardware_**|Physical machines.
|**_HCC (Heterogeneous Compute Compiler) (Deprecated)_**|An Open Source, Optimizing C++ Compiler for Heterogeneous Compute. It supports heterogeneous offload to AMD APUs and discrete GPUs via HSA enabled runtimes and drivers. It is based on Clang, the LLVM Compiler Infrastructure and the 'libc++' C++ standard library. The goal is to implement a compiler that takes a program that conforms to a parallel programming standard such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP, and transforms it into the AMD GCN ISA.
|**_hidden layer_**|The layer between input layers and output layers. Artificial neurons intake inputs and produce an output.
|**_HIP (Heterogeneous Interface for Portability)_**|Heterogeneous Interface for Portability is a C++ runtime API and kernel language that allows developers to create portable applications that can run on AMD and other GPU's. It provides a C-style API and a C++ kernel language. The first big feature available in the HIP is porting apps that use the CUDA Driver API.
|**_HIP-Clang_**|Heterogeneous AMDGPU Compiler, with its capability to compile HIP programs on AMD platform
|**_hipconfig_**|Tool to report various configuration properties of the target platform.
|**_hipify tools_**|Tools to convert CUDA® code to portable C++ code.
|**_hipSetDevice_**|A set default device for hip API calls.
|**_host, host cpu_**|Executes the HIP runtime API and is capable of initiating kernel launches to one or more devices.
|**_HPC (High Performance Computing)_**|The ability to perform and process data at high speeds.
|**_HSA (Heterogeneous System Architecture)_**|HSA provides a unified view of fundamental computing elements. HSA allows a programmer to write applications that seamlessly integrate CPUs (called latency compute units) with GPUs (called throughput compute units), while benefiting from the best attributes of each. HSA creates an improved processor design that exposes the benefits and capabilities of mainstream programmable compute elements, working together seamlessly. HSA is all about delivering new, improved user experiences through advances in computing architectures that deliver improvements across all four key vectors: improved power efficiency; improved performance; improved programmability; and broad portability across computing devices. For more on HSA.(http://developer.amd.com/wordpress/media/2012/10/hsa10.pdf)
|**_HTC (High Throughput Computing)_**|Allows system to run multiple software across multiple processors and the same time. Refers to the computing over long periods of time.
|**_hyperparameter_**|Values that control learning process using learning algorithm.
|**_image recognition_**|The systems or software ability to identify objects or details in an image.
|*_imbalanced dataset_**|Datasets where the target class is unequally distributed.
|**_inference_**|Running live data points into a machine learning algorithm.
|**_Java, JavaScript, JScript_**|An object-oriented programming language.
|**_Jira_**|A bug tracking and agile project management software.
|**Keras**|An open-source library that acts as an interface for TensorFlow library.
|**_Kernel_**|Program working within the operating system that controls the system. Facilitates interactions between hardware and software.
|**_Kernel GPU Driver_**|Software program that runs within the OS that between the applications, and graphics card components.
|**_KVM_**|Kernel-based Virtual Machine, an open-source virtualization within Linux that allows a host of machine to run multiple virtual environments.
|**_learning rate_**|The hyperparameter that controls the rate/speed that a model learns.
|**_Linux_**|An open-source OS, that manages the system's hardware, CPU, memory, and storage.
|**_Long Short-Term Memory (LSTM)_**|A recurrent neural network that processes data long-term.
|**_machine learning_**|The use and development of computer systems with the capability of learning and adapting using algorithms.
|**_matplotlib_**|An open-source library within Python that visualizes data.
|**_memory_**|Refer to specific type. RAM, ROM, hard drive, local, etc.
|**_method (function)_**|An action performed by the API. The method resides within the API.
|**_micro-architecture_**|The specific design of a microprocessor.
|**_microprocessor_**|A CPU on a single IC.
|**_MIOpen_**|AMD’s deep learning primitive’s library which provides highly optimized, and hand-tuned implementations of different operators such as convolution, batch normalization, pooling, softmax, activation and layers for Recurrent Neural Networks (RNNs), used in both training and inference.
|**_MLPerf_**|Utilized for measuring the speed in which systems can run models in different scenarios.
|**_MNIST_**|A large database of handwritten digits used for training image processing systems.
|**_model training_**|The phase in development where datasets are used to train the Machine Learning algorithm.
|**_neural network_**|A network of functions used to translate data input into a desire output.
|**_neuron_**|A function that takes a group of inputs and process them to be the output for the next layer. Its purpose is to model the actual functioning of a biological neuron.
|**_node (neural network)_**|Neuron in a hidden layer.
|**_node (TensorFlow graph)_**|An operation in TensorFlow graph.
|**_nodes_**|A unit that has one or more weighted input connections. See: neuron.
|**_normalization_**|The reorganization of data into the same scale.
|**_null_**|Null value
|**_NumPy_**|A linear algebra library in Python, it supports mutli-dimensional arrays and matrices.
|**_nvcc_**|nvcc compiler, do not capitalize.
|**_offline inference_**|The process of generating predictions on batch of observations that occur on a recurring schedule.
|**_one-hot encoding_**|The method of converting data into a form for Machine Learning algorithms.
|**_OpenACC_**|For accelerators, a programming standard for parallel computing on accelerators.
|**_OpenCL_**|Open Computing Language (OpenCL) is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators. OpenCL provides a standard interface for parallel computing using task- and data-based parallelism. The programming language that is used to write compute kernels is called OpenCL C and is based on C99,[16] but adapted to fit the device model in OpenCL. OpenCL consists of a set of headers and a shared object that is loaded at runtime. As of 2016 OpenCL runs on Graphics processing units, CPUs with SIMD instructions, FPGAs, Movidius Myriad 2, Adapteva epiphany and DSPs.
|**_OpenMP_**|Programming model, supports offloading code and data onto GPU devices.
|**_Open MPI_**|A Message Passing Interface (MPI) library project.
|**_Open-source_**|Software for which the source code is publicly accessible.
|**_optimizer_**|A program that utilizes linear programming to optimize processes.
|**_outliers_**|Values that are abnormal.
|**_overfitting_**|A model learns data to the extent that it negatively impact performance of the model on new data.
|**_oversampling_**|Duplicate examples from minority class. Data analysis technique that adjusts the class distribution of a dataset.
|**_Pandas_**|An open-source data analysis tool built on top of Python programing language.
|**_Parameters_**|Displayed in parentheses when written in a function, parameters act as variables that assign value.
|**_PCIe Platform Atomics_**|PCI Express (PCIe) was developed as the next generation I/O system interconnect after PCI, designed to enable advanced performance and features in connected devices while remaining compatible with the PCI software environment. Today, atomic transactions are supported for synchronization without using an interrupt mechanism. In emerging applications where math co-processing, visualization and content processing are required, enhanced synchronization would enable higher performance.
|**_perceptron_**|A mathematical model of a biological neuron.
|**_PerfZero_**|Framework for TensorFlow. "...consolidating the docker image build, GPU driver installation, TensorFlow installation, benchmark library checkout, data download, system statistics collection, benchmark metrics collection, profiler data collection and so on into 2 to 3 commands. "
|**_precision_**|The quality of positive predictions made.
|**_precision-recall curve_**|The performance metrics between precision and recall for binary classification algorithms.
|**_prediction_**|The output of an algorithm on a set of data based on predictive analytics.
|**_prediction bias_**|The difference between a model apparent and prediction error.
|**_pre-training model_**|A model created based on already created and trained models.
|**_probabilistic regression model_**|Predictions made on a dependent variable based on information
|**_PyTorch_**|An open-source machine learning library used for Python.
|**_Queue_**|A Queue is a runtime-allocated resource that contains a packet buffer and is associated with a packet processor. The packet processor tracks which packets in the buffer have already been processed. When it has been informed by the application that a new packet has been enqueued, the packet processor is able to process it because the packet format is standard, and the packet contents are self-contained -- they include all the necessary information to run a command. A queue has an associated set of high-level operations defined in "HSA Runtime Specification" (API functions in host code) and "HSA Programmer Reference Manual Specification" (kernel code).
|**_rank (Tensor)_**|A unit of dimensionality, the number of dimensions of the tensor within the TensorFlow system.
|**_RCCL_**|A stand-alone library of standard collective communication routines for GPUs.
|**_recall_**|The true positives found. The capability for a model to find relevant cases in a dataset.
|**_recommendation system_**|A system designed for predicting user responses.
|**_Rectified Linear Unit (ReLU)_**|A non-linear activation function that will output the input directly if it is positive.
|**_recurrent neural network_**|An artificial neural network set to work for time series data.
|**_regression model_**|A statistical analysis for estimate the connection between outcome variable and predictors.
|**_regularization_**|A form of regression reduces error by forming a function on the training set.
|**_regularization rate_**|A scalar value that specifies the relative importance of regularization function. Represented as lambda.
|**_representation_**|The process of mapping data to useful features.
|**_REST API_**|A style of networked applications limited to a client-server-based applications.
|**_RBT_**|Risk-Based Testing
|**_RHEL_**|A subscription-based operating system. You must enable the external repositories to install on the devtoolset-7 environment and the dkms support files.
|**_RNG, FFT_**|The random number generator and fast fourier transform algorithms have many applications in creating GPU applications for different fields.
|**_ROCclr_**|A virtual device interface that compute runtimes interact with different backends such as ROCr on Linux or PAL on Windows. The ROCclr (https://github.com/ROCm-Developer-Tools/ROCclr) is an abstraction layer allowing runtimes to work on both OSes without much effort.(https://github.com/ROCm-Developer-Tools/ROCclr)
|**_ROCdbgapi_**|The AMD Debugger API is a library that provides all the support necessary for a debugger and other tools to perform low level control of the execution and inspection of execution state of AMD's commercially available GPU architectures.
|**_rocFFT_**|An implementation of Fast Fourier Transform (FFT) written in HIP for GPUs.
|**_rocGDB_**|ROCm source-level debugger for Linux.
|**_ROC (receiver operating characteristic) Curve_**|A curve of true positive rate vs. false positive rate at different classification thresholds.
|**_ROCm_**| ROCm is a brand name for ROCm™ open software platform (for software) or the ROCm™ open platform ecosystem (includes hardware like FPGAs or other CPU architectures.) **_NOTE_**: ROCm no longer functions as an acronym.
|**_ROCm SMI_**|ROCm System Management Interface, command line interface for manipulating and monitoring the amdgpu kernel.
|**_ROCm Validation Suite_**|The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager’s tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.
|**_rocProfiler_**|A command line tool used for API environment configurations. For profiling GPU compute applications.
|**_rocSOLVER_**|An implementation of LAPACK routines on top of the AMD’s open source ROCm platform. rocSOLVER is implemented in the HIP programming language and optimized for AMD’s latest discrete GPUs.
|**_rocSPARSE_**|Exposes a common interface that provides Basic Linear Algebra Subroutines for sparse computation implemented on top of AMD’s Radeon Open Compute ROCm runtime and toolchains. rocSPARSE is created using the HIP programming language and optimized for AMD’s latest discrete
|**_ROCr ROCm runtime_**|The HSA runtime is a thin, user-mode API that exposes the necessary interfaces to access and interact with graphics hardware driven by the AMDGPU driver set and the ROCK kernel driver. Together they enable programmers to directly harness the power of AMD discrete graphics devices by allowing host applications to launch compute kernels directly to the graphics hardware. **Accelerator Modes Supported**: HC C++ API, HIP, C++ AMP, C++ Parallel STL, OpenMP
|**_ROCT-THUNK/ROC Runtime_**|A cmake-based system available for building thunk.
|**_rocThrust_**|A parallel algorithm library, runs Thrust dependent software on AMD GPUs.
|**_rocTracer_**|Runtime API calls and asynchronous GPU activity tracer.
|**_Runtime_**|The execution of a program.
|**_sampling bias_**|Collected data does not accurately reflect the program expected to run.
|**_scalar_**|Consisting of a single value (e.g. integer or string) rather than multiple values (e.g. array).
|**_scaling_**|The expansion of configuration to handle/manage the load amount on the server.
|**_scikit-learn_**|Python's machine learning library, features clustering algorithms, regression, and classification.
|**_scoring_**|The action of building an algorithmic model from historical dataset to a new dataset.
|**_script_**|Embedded code.
|**_SDK_**|Software development kit, used to implement API.
|**_selection bias_**|When a dataset is not reflective of reality of the environment the model runs.
|**_self-supervised learning_**|A machine learning technique where the model learns by using unlabeled data to predict and generate missing information.
|**_self-training_**|See: semi-training
|**_semi-supervised learning_**|An approach that trains using a small amount of label data and the model makes predictions of unlabeled data. A combination of supervised and unsupervised training.
|**_Shader Engines_**|A program that tells the model how to render pixels.
|**_shape (tensor)_**|The number of elements in dimensions.
|**_Source code_**| Code written by developers for a computer program.
|**_Sparse_**|Some problems are defined in terms of linear operations over arrays of data (e.g. vectors and matrices) the elements of which are mostly zeros (sparse arrays). When the fraction of zeros is significantly large, enough so that there are benefits to explicitly take these zeros into account when solving the problem, these problems are called Sparse Linear Algebra problems.
|**_squared loss_**|A function that is used to indicate how much predicted output is off.
|**_staged training_**|A training method for a model that forms in stages.
|**_supervised machine learning_**|Training model that utilizes input data and corresponding labels.
|**_tensor processing unit_**|Google's custom application-specific integrated circuits that is used to accelerate machine learning workloads.
|**_tensor shape_**|Number of elements a Tensor contains in various dimension.
|**_tensor size_**|Number of scalars a Tensor contains.
|**_TensorFlow_**|An end-to-end open-source platform used for machine learning, contains a system of tools, libraries and community resources that allows developers to build machine learning powered applications
|**_tensorflow playground_**|An interactive visualization of neural networks.
|**_time series_**|The sequence of observations collected in constant time intervals.
|**_token_**|An instance of a sequence of characters in some document that grouped together as a useful semantic unit for processing.
|**_training_**|The process of learning a machine learning model using algorithms and training data.
|**_training set_**|A set of data examples provided during the learning process.
|**_transformer_**|A model architecture for transforming one sequence into another using Encoder and Decoder.
|**_true negative_**|An output where the model predicts a negative class correctly.
|**_true positive_**|An output where the model predicts a positive class correctly.
|**_Ubuntu_**|An open-source Linux distribution
|**_unsupervised machine learning_**|Utilizing algorithms to analyze unlabeled dataset. The algorithm detects hidden patterns and data groupings without a human user.
|**_upstream_**|Reference to the flow of open-source code. If code developed from an origin however, it is incorporated into original source code.
|**_validation_**|The use of a testing data set to evaluate a training model.
|**_vanishing gradient_**|When useful gradient information is unable to be propagated from the output end of the model back to the layers near the input end of the model.
|**_variable_**|Assigned within the function and carries data input throughout the function.     			 |
