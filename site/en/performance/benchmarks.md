# Benchmarks

## Overview

A selection of image classification models were tested across multiple platforms
to create a point of reference for the TensorFlow community. The
[methodology](#methodology) section details how the tests were executed and has
links to the scripts used.

## Results for image classification

ResNet-50
([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
was tested using the [ImageNet](http://www.image-net.org/) data set. Tests were
run on Google Compute Engine, Amazon Elastic Compute Cloud (Amazon EC2), and an
NVIDIA® DGX-1™. Most of the tests were run with both real data and symthetic data.
Real data represents the experience doing real work. Synthetic data is useful
for quick tests to verify hardware without having to downloaded and prepare a
dataset as well as to eliminate disk I/O as a variable and to set a baseline.
Testing with synthetic data was done by using a `tf.Variable` set to the same
shape as the data expected by the model. The `tf.Variable` is 
placed directly on the device avoiding a host to device copy.

For most of the tests below we executed both ResNetV1 and ResNetV1.5.
ResNetV1 has been the historical benchmark for the past few years. In the past
year a varient of V1 has been come popular but not named where the first conv 3x3
stride something something is different. The MLPerf group is calling this
variant V1.5.  V1.5 whie often just refered to as ResNet has become the most
popular variant.  V1.5 is more stable at higher batch sizes and trains to a higher
accuracy than the original V1 given the same number of epochs. For a detailed
break down of V1 vs. V1.5 see the section blah blah.

Total training time can be reduced by reducing the by reducing the number
of epochs by adjusting hyperparameters or utilizing new techniques. These benchmarks
are about providing information about the behavior of TensorFlow on various hardware
platforms keeping the code consistent.

Accuracy of models tested
INESRT TABLE  81 epochs ResNet50V1.5  90 ResNetV1 and 90 ResNetV1.5   FP16 and FP32 
talk about V1 ranging if we want because it does top_1 and top_5 results desired 
just on one platform not sharing total time as that is a distraction.



### Training with NVIDIA® DGX-1™ (NVIDIA® Tesla® V100SMX2)

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="../images/perf_summary_p100_single_server.png">
</div>

Details and additional results are in the [Details for NVIDIA® DGX-1™ (NVIDIA®
Tesla® P100)](#details_for_nvidia_dgx-1tm_nvidia_tesla_p100) section.

### Training with NVIDIA® DGX-1™ (NVIDIA® Tesla® P100)

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="../images/perf_summary_p100_single_server.png">
</div>

Details and additional results are in the [Details for NVIDIA® DGX-1™ (NVIDIA®
Tesla® P100)](#details_for_nvidia_dgx-1tm_nvidia_tesla_p100) section.

### Training with NVIDIA® Tesla® K80

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="../images/perf_summary_k80_single_server.png">
</div>

Details and additional results are in the [Details for Google Compute Engine
(NVIDIA® Tesla® K80)](#details_for_google_compute_engine_nvidia_tesla_k80) and
[Details for Amazon EC2 (NVIDIA® Tesla®
K80)](#details_for_amazon_ec2_nvidia_tesla_k80) sections.

### Inference with NVIDIA® Tesla® P4 and TensorRT


### FP32 vs FP16 (mixed-precision)
Talk about FP32 and FP16 and show some graphs as well as what hardware benefits the change.

### ResNetV1 vs ResNetV1.5

## Details for NVIDIA® DGX-1™ (NVIDIA® Tesla® V100SMX2)

### Environment

*   **Instance type**: NVIDIA® DGX-1™
*   **GPU:** 8x NVIDIA® Tesla® P100
*   **OS:** Ubuntu 16.04 LTS with tests run via Docker
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** Local SSD
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

Commands used for test:

```bash

Put the commands here.  

```


### Results

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="../images/perf_summary_p100_single_server.png">
</div>

**Training real data**

GPUs | ResNet-50 v1 | ResNet-50 v1.5 | Trivial  
---- | ------------ | -------------- | --------   
1    | 142          | 219            | 91.8       
2    | 284          | 422            | 181        
4    | 569          | 852            | 356        
8    | 1131         | 1734           | 716  

**Training synthetic data**

GPUs | ResNet-50 v1 | ResNet-50 v1.5 
---- | ------------ | --------------  
1    | 142          | 219                  
2    | 284          | 422                    
4    | 569          | 852                   
8    | 1131         | 1734            


## Details for NVIDIA® DGX-1™ (NVIDIA® Tesla® P100)

### Environment

*   **Instance type**: NVIDIA® DGX-1™
*   **GPU:** 8x NVIDIA® Tesla® P100
*   **OS:** Ubuntu 16.04 LTS with tests run via Docker
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** Local SSD
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

Commands used for test:

```bash

Put the commands here.  

```

### Results

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="../images/perf_summary_p100_single_server.png">
</div>

**Training real data**

GPUs | ResNet-50 v1 | ResNet-50 v1.5 | Trivial  
---- | ------------ | -------------- | --------   
1    | 142          | 219            | 91.8       
2    | 284          | 422            | 181        
4    | 569          | 852            | 356        
8    | 1131         | 1734           | 716  

**Training synthetic data**

GPUs | ResNet-50 v1 | ResNet-50 v1.5 
---- | ------------ | --------------  
1    | 142          | 219                  
2    | 284          | 422                    
4    | 569          | 852                   
8    | 1131         | 1734  

## Details for Google Cloud (NVIDIA® Tesla® V100SMX2)

### Environment

*   **Instance type**: NVIDIA® DGX-1™
*   **GPU:** 8x NVIDIA® Tesla® P100
*   **OS:** Ubuntu 16.04 LTS with tests run via Docker
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** Local SSD
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

Commands used for test:

```bash

Put the commands here.  

```

### Results

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="../images/perf_summary_p100_single_server.png">
</div>

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="../images/perf_dgx1_synth_p100_single_server_scaling.png">
  <img style="width:35%" src="../images/perf_dgx1_real_p100_single_server_scaling.png">
</div>

**Training real data**

GPUs | ResNet-50 v1 | ResNet-50 v1.5 | Trivial  
---- | ------------ | -------------- | --------   
1    | 142          | 219            | 91.8       
2    | 284          | 422            | 181        
4    | 569          | 852            | 356        
8    | 1131         | 1734           | 716  

**Training synthetic data**

GPUs | ResNet-50 v1 | ResNet-50 v1.5 
---- | ------------ | --------------  
1    | 142          | 219                  
2    | 284          | 422                    
4    | 569          | 852                   
8    | 1131         | 1734  

## Details for Amazon EC2 (NVIDIA® Tesla® K80)

### Environment

*   **Instance type**: p2.8xlarge
*   **GPU:** 8x NVIDIA® Tesla® K80
*   **OS:** Ubuntu 16.04 LTS
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** 1TB Amazon EFS (burst 100 MiB/sec for 12 hours, continuous 50
    MiB/sec)
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

Commands used for test:

```bash

Put the commands here.  

```

### Results

**Training real data**

GPUs | ResNet-50 v1  
---- | ------------  
1    | 142           
2    | 284           
4    | 569           
8    | 1131         

**Training synthetic data**

GPUs | ResNet-50 v1  
---- | ------------ 
1    | 142                    
2    | 284                     
4    | 569                  
8    | 1131         


## Methodology

This
[script](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)
was run on the various platforms to generate the above results.  Check each test for the
exact hash point that was used.

In order to create results that are as repeatable as possible, each test was run
5 times and then the times were averaged together. GPUs are run in their default
state on the given platform. For NVIDIA® Tesla® K80 this means leaving on [GPU
Boost](https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/).
For each test, 10 warmup steps are done and then the next 100 steps are
averaged.

Both Google Cloud and Amazon's AWS offer Deep Learning VMs with TensorFlow compiled
with flags optimized for their systems and at times newer versions of NVIDIA libraries.
For consistency the benchmarks are run via Docker using TensorFlow's official Docker
images.
