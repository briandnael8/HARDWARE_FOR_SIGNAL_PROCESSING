# **HARDWARE_FOR_SIGNAL_PROCESSING**

This project implements the inference phase of the LeNet-5 Convolutional Neural Network (CNN) on a GPU using CUDA. The implementation focuses on leveraging the GPU for parallel computations to achieve significant acceleration compared to a CPU-based approach.

## **Project Objectives**

- Learn and use CUDA for GPU programming.
- Analyze algorithmic complexity and compare GPU vs. CPU performance.
- Implement LeNet-5 inference from scratch (excluding training).
- Export weights trained using PyTorch and import them into the CUDA program.
- Utilize Git for version control and collaboration.

## **Overview of LeNet-5**

LeNet-5, proposed by Yann LeCun in 1998, is a classical CNN architecture designed for image classification tasks, such as handwritten digit recognition using the MNIST dataset. It consists of:
1. Convolutional and subsampling layers.
2. Fully connected layers.
3. A final classification layer using softmax.

More information can be found in the [original paper](https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).

---

## **Implemented Features**

### **1. CUDA Basics**
- **Matrix Operations**:
  - CPU implementations: MatrixInit, MatrixAdd, MatrixPrint.
  - GPU implementations: cudaMatrixAdd, cudaMatrixMult.
  - Performance comparison: Demonstrated speedup on GPU.

### **2. LeNet-5 Inference**
#### Layers Implemented:
1. **Input Layer**:
   - Input data initialized as a 32x32 matrix with random values.
2. **Convolution Layer**:
   - Applied 6 convolution kernels (5x5).
   - Output: 6 feature maps of size 28x28.
3. **Subsampling Layer**:
   - Downsampling by averaging 2x2 pixel blocks.
   - Output: 6 feature maps of size 14x14.
4. **Activation Function**:
   - Added non-linearity with the tanh activation function.
5. **Fully Connected Layers**:
   - Combined feature maps for classification.

#### Dataset:
- **MNIST**:
  - Data processed in binary format.
  - Conversion to 28x28 matrices.
- Exported weights trained using PyTorch for the final inference.

---

## **Performance Analysis**
- **Matrix Multiplication**:
  - Complexity: \(O(n^3)\)
  - Acceleration: Achieved a **6.2x speedup** on GPU compared to CPU for larger matrices.
- **CNN Layers**:
  - Measured GPU execution time:
    - Convolution + Subsampling: **3.24 ms**.
  - Observations:
    - GPU excels for large-scale parallelism.
    - Performance limited for smaller matrices.

---

## **Challenges & Limitations**
- **Weight Import Issues**:
  - Some weights exported from PyTorch were incorrectly retrieved as zeros.
  - This impacted the final prediction accuracy, as convolutions and fully connected layers were compromised.
- **GPU Limitations**:
  - Inefficient for small-scale computations or low parallelism.

---

## **Usage Instructions**

### **Dependencies**
- **CUDA Toolkit**: For GPU programming.
- **PyTorch**: For training the model and exporting weights.
- **MNIST Dataset**: Preprocessed binary files.

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LeNet5-CUDA.git
   cd LeNet5-CUDA
