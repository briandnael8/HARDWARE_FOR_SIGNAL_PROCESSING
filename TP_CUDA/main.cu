// NAEL BRIAND & QUENTIN DE LA CHAISE


/////////////////////////////////////////////////////////////////////////////////////
// Include /////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iostream>
#include <cstdlib> 
#include <ctime>    
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// PARTIE 1 : Getting started with CUDA: Matrix multiplication /////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
// Definition of CPU functions /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%0.2f ", M[i * p + j]);
        }
        printf("\n");
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// Definition of GPU functions /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        int idx = row * p + col;
        Mout[idx] = M1[idx] + M2[idx];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0;
        for (int k = 0; k < n; k++) {
            value += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = value;
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// Definition of measure time functions ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

double measureTimeCPU(int n) {
    float *M1 = (float *)malloc(n * n * sizeof(float));
    float *M2 = (float *)malloc(n * n * sizeof(float));
    float *Mout = (float *)malloc(n * n * sizeof(float));

    MatrixInit(M1, n, n);
    MatrixInit(M2, n, n);

    clock_t start = clock();
    MatrixMult(M1, M2, Mout, n);
    clock_t end = clock();

    double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    free(M1);
    free(M2);
    free(Mout);

    return cpuTime;
}

double measureTimeGPU(int n) {
    float *M1, *M2, *Mout;

    cudaMalloc(&M1, n * n * sizeof(float));
    cudaMalloc(&M2, n * n * sizeof(float));
    cudaMalloc(&Mout, n * n * sizeof(float));

    float *h_M1 = (float *)malloc(n * n * sizeof(float));
    float *h_M2 = (float *)malloc(n * n * sizeof(float));
    MatrixInit(h_M1, n, n);
    MatrixInit(h_M2, n, n);
    cudaMemcpy(M1, h_M1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(M2, h_M2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    cudaMatrixMult<<<blocksPerGrid, threadsPerBlock>>>(M1, M2, Mout, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(M1);
    cudaFree(M2);
    cudaFree(Mout);
    free(h_M1);
    free(h_M2);

    return milliseconds / 1000.0;
}

/////////////////////////////////////////////////////////////////////////////////////
// Testing functions to see influences of matrix size //////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

void testMatrixSizes() {
    std::vector<int> sizes = {200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};
    std::vector<double> cpuTimes, gpuTimes;

    for (int n : sizes) {
        std::cout << "Testing size: " << n << "x" << n << std::endl;
        double cpuTime = measureTimeCPU(n);
        double gpuTime = measureTimeGPU(n);

        cpuTimes.push_back(cpuTime);
        gpuTimes.push_back(gpuTime);

        std::cout << "CPU Time: " << cpuTime << " seconds, GPU Time: " << gpuTime << " seconds" << std::endl;
    }

    std::ofstream file("results.csv");
    file << "Size,CPU,GPU\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        file << sizes[i] << "," << cpuTimes[i] << "," << gpuTimes[i] << "\n";
    }
    file.close();
}

/////////////////////////////////////////////////////////////////////////////////////
// MAIN ////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

//// Testing initialization, multiplication and addition functions
// int main() {
//     int n = 3;
//     int p = 3;

//     float *M1 = (float *)malloc(n * p * sizeof(float));
//     float *M2 = (float *)malloc(n * p * sizeof(float));
//     float *Mout = (float *)malloc(n * p * sizeof(float));

//     MatrixInit(M1, n, p);
//     MatrixInit(M2, n, p);

//     printf("Matrice 1 :\n");
//     MatrixPrint(M1, n, p);
//     printf("\nMatrice 2 :\n");
//     MatrixPrint(M2, n, p);

//     printf("\nAddition on CPU :\n");
//     MatrixAdd(M1, M2, Mout, n, p);
//     MatrixPrint(Mout, n, p);

//     printf("\nMultiplication on CPU :\n");
//     MatrixMult(M1, M2, Mout, n);
//     MatrixPrint(Mout, n, p);

//     free(M1);
//     free(M2);
//     free(Mout);

//     return 0;
// }

// int main() {
//     int n = 50; 

//     std::cout << "Mesure des performances CPU..." << std::endl;
//     measureTimeCPU(n);

//     std::cout << "\nMesure des performances GPU..." << std::endl;
//     measureTimeGPU(n);

//     return 0;
// }

// int main() {
//     
//     std::cout << "Début des tests de performances CPU/GPU pour la multiplication de matrices...\n";
//     testMatrixSizes();
//     std::cout << "Tests terminés. Les résultats sont enregistrés dans 'results.csv'.\n";

//     return 0;
// }

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// PARTIE 2 : First layers of the LeNet-5 neural network: 2D convolution and subsampling 
///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////
// Initializing matrices ///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

// We chose the first option for the Matrix Initialization :
// 1-dimensional arrays N=32x32, 6x28x28, 6x14x14 and 6x5x5 respectively, where each cell corresponds to one element.

void initializeRawData(float *raw_data) {
    for (int i = 0; i < 32 * 32; i++) {
        raw_data[i] = ((float)rand() / RAND_MAX); 
    }
}


void initializeC1Data(float *C1_data) {
    for (int i = 0; i < 6 * 28 * 28; i++) {
        C1_data[i] = 0.0f; 
    }
}

void initializeS1Data(float *S1_data) {
    for (int i = 0; i < 6 * 14 * 14; i++) {
        S1_data[i] = 0.0f; 
    }
}

void initializeC1Kernel(float *C1_kernel) {
    for (int i = 0; i < 6 * 5 * 5; i++) {
        C1_kernel[i] = ((float)rand() / RAND_MAX); // Valeurs entre -1 et 1
    }
}


/////////////////////////////////////////////////////////////////////////////////////
// Convolution 2D function /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

// __global__ void convolution2D(float *input, float *output, float *kernel, int depth, int in_width, int out_width, int kernel_size) {
//     int tx = threadIdx.x + blockIdx.x * blockDim.x;
//     int ty = threadIdx.y + blockIdx.y * blockDim.y;

//     if (tx < out_width && ty < out_width) {
//         int output_idx = depth * out_width * out_width + ty * out_width + tx; // Index pour la sortie
//         float sum = 0.0f;

//         for (int ky = 0; ky < kernel_size; ky++) {
//             for (int kx = 0; kx < kernel_size; kx++) {
//                 int input_x = tx + kx;
//                 int input_y = ty + ky;
//                 int input_idx = input_y * in_width + input_x; // Index pour l'entrée (pas de profondeur pour raw_data)
//                 int kernel_idx = depth * kernel_size * kernel_size + ky * kernel_size + kx; // Index pour le filtre
//                 sum += input[input_idx] * kernel[kernel_idx];
//             }
//         }

//         output[output_idx] = sum;
//     }
// }

/////////////////////////////////////////////////////////////////////////////////////
// Subsampling function ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

__global__ void subsampling(float *input, float *output, int depth, int in_width, int out_width) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < out_width && ty < out_width) {
        int output_idx = depth * out_width * out_width + ty * out_width + tx; // Index pour la sortie
        int input_idx = depth * in_width * in_width + (ty * 2) * in_width + (tx * 2); // Index pour l'entrée

        float sum = 0.0f;
        sum += input[input_idx];
        sum += input[input_idx + 1];
        sum += input[input_idx + in_width];
        sum += input[input_idx + in_width + 1];

        output[output_idx] = sum / 4.0f; // Moyenne des 4 pixels
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// Activation function /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

__device__ float activation_tanh(float M) {
    return tanhf(M); 
}

/////////////////////////////////////////////////////////////////////////////////////
// Convolution 2D function with activation function ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

// __global__ void convolution2D(float *input, float *output, float *kernel, int depth, int in_width, int out_width, int kernel_size) {
//     int tx = threadIdx.x + blockIdx.x * blockDim.x;
//     int ty = threadIdx.y + blockIdx.y * blockDim.y;

//     if (tx < out_width && ty < out_width) {
//         int output_idx = depth * out_width * out_width + ty * out_width + tx; // Index pour la sortie
//         float sum = 0.0f;

//         for (int ky = 0; ky < kernel_size; ky++) {
//             for (int kx = 0; kx < kernel_size; kx++) {
//                 int input_x = tx + kx;
//                 int input_y = ty + ky;
//                 int input_idx = input_y * in_width + input_x; // Index pour l'entrée
//                 int kernel_idx = depth * kernel_size * kernel_size + ky * kernel_size + kx; // Index pour le filtre
//                 sum += input[input_idx] * kernel[kernel_idx];
//             }
//         }

//         // Normaliser la somme pour éviter la saturation de tanh
//         sum /= (float)(kernel_size * kernel_size); // Divise par le nombre total d'éléments dans le kernel
//         output[output_idx] = activation_tanh(sum);
//     }
// }

/////////////////////////////////////////////////////////////////////////////////////
// MAIN ////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


// // Testing the matrix initialization

// int main() {
//     // Initialiser le générateur de nombres aléatoires
//     srand(time(0));

//     // Allocation de mémoire pour les matrices
//     float *raw_data = (float *)malloc(32 * 32 * sizeof(float)); // Données d'entrée
//     float *C1_data = (float *)malloc(6 * 28 * 28 * sizeof(float)); // Sortie convolution
//     float *S1_data = (float *)malloc(6 * 14 * 14 * sizeof(float)); // Sortie subsampling
//     float *C1_kernel = (float *)malloc(6 * 5 * 5 * sizeof(float)); // Filtres convolution

//     // Initialisation des matrices
//     initializeRawData(raw_data);
//     initializeC1Data(C1_data);
//     initializeS1Data(S1_data);
//     initializeC1Kernel(C1_kernel);

//     // Vérification des données initialisées
//     std::cout << "Premier élément de raw_data : " << raw_data[0] << std::endl;
//     std::cout << "Premier élément de C1_data : " << C1_data[0] << std::endl;
//     std::cout << "Premier élément de S1_data : " << S1_data[0] << std::endl;
//     std::cout << "Premier élément de C1_kernel : " << C1_kernel[0] << std::endl;

//     // Affichage de quelques valeurs de raw_data pour valider
//     std::cout << "\nQuelques éléments de raw_data : ";
//     for (int i = 0; i < 5; i++) {
//         std::cout << raw_data[i] << " ";
//     }
//     std::cout << std::endl;

//     // Affichage de quelques valeurs de C1_kernel pour valider
//     std::cout << "\nQuelques éléments de C1_kernel : ";
//     for (int i = 0; i < 5; i++) {
//         std::cout << C1_kernel[i] << " ";
//     }
//     std::cout << std::endl;

//     // Libération de la mémoire
//     free(raw_data);
//     free(C1_data);
//     free(S1_data);
//     free(C1_kernel);

//     return 0;
// }

// // TESTS : First layers of the CNN with "simple" values.

// int main() {
//     // Allocation de mémoire sur CPU
//     // float *raw_data = (float *)malloc(32 * 32 * sizeof(float));      // Données d'entrée
//     // float *C1_data = (float *)malloc(6 * 28 * 28 * sizeof(float));   // Sortie convolution
//     // float *S1_data = (float *)malloc(6 * 14 * 14 * sizeof(float));   // Sortie subsampling
//     // float *C1_kernel = (float *)malloc(6 * 5 * 5 * sizeof(float));   // Filtres convolution

//     // Initialisation des données avec des valeurs simples
// //     for (int i = 0; i < 32 * 32; i++) {
// //     if (i % 3 == 0) {
// //         raw_data[i] = 1.0f; // Une partie des données est initialisée à 1
// //     } else if (i % 3 == 1) {
// //         raw_data[i] = 0.5f; // Une autre partie des données est initialisée à 0.5
// //     } else {
// //         raw_data[i] = 0.0f; // Le reste est initialisé à 0
// //     }
// // }

//     //Allocation de mémoire pour les matrices
//     float *raw_data, *C1_data, *C1_kernel, *S1_data;
//     cudaMallocManaged(&raw_data, 32 * 32 * sizeof(float));
//     cudaMallocManaged(&C1_data, 6 * 28 * 28 * sizeof(float));
//     cudaMallocManaged(&S1_data, 6 * 14 * 14 * sizeof(float));
//     cudaMallocManaged(&C1_kernel, 6 * 5 * 5 * sizeof(float));

//     // Initialisation des données
//     initializeRawData(raw_data);
//     initializeC1Data(C1_data);
//     initializeC1Kernel(C1_kernel);
//     initializeS1Data(S1_data);
//     // for (int i = 0; i < 6 * 5 * 5; i++) {
//     //     C1_kernel[i] = 1.0f; // Toutes les valeurs des filtres sont égales à 1
//     // }
//     // for (int i = 0; i < 6 * 28 * 28; i++) {
//     //     C1_data[i] = 0.0f; // Initialisation à 0
//     // }
//     // for (int i = 0; i < 6 * 14 * 14; i++) {
//     //     S1_data[i] = 0.0f; // Initialisation à 0
//     // }

//     // Afficher raw_data avant la convolution
//     printf("Matrice d'entrée (raw_data) :\n");
//     MatrixPrint(raw_data, 32, 32);

//     // Allocation de mémoire sur GPU
//     float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;
//     cudaMalloc(&d_raw_data, 32 * 32 * sizeof(float));
//     cudaMalloc(&d_C1_data, 6 * 28 * 28 * sizeof(float));
//     cudaMalloc(&d_S1_data, 6 * 14 * 14 * sizeof(float));
//     cudaMalloc(&d_C1_kernel, 6 * 5 * 5 * sizeof(float));

//     // Copier les données CPU vers GPU
//     cudaMemcpy(d_raw_data, raw_data, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_C1_kernel, C1_kernel, 6 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);

//     // Appliquer la convolution
//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((28 + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (28 + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     for (int d = 0; d < 6; d++) {
//         convolution2D<<<blocksPerGrid, threadsPerBlock>>>(d_raw_data, d_C1_data, d_C1_kernel, d, 32, 28, 5);
//     }

//     cudaDeviceSynchronize();

//     // Copier C1_data sur CPU et afficher
//     cudaMemcpy(C1_data, d_C1_data, 6 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);
//     printf("\nMatrice après convolution (C1_data, profondeur 0) :\n");
//     MatrixPrint(C1_data, 28, 28);

//     // Appliquer le sous-échantillonnage
//     blocksPerGrid = dim3((14 + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                          (14 + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     for (int d = 0; d < 6; d++) {
//         subsampling<<<blocksPerGrid, threadsPerBlock>>>(d_C1_data, d_S1_data, d, 28, 14);
//     }

//     cudaDeviceSynchronize();

//     // Copier S1_data sur CPU et afficher
//     cudaMemcpy(S1_data, d_S1_data, 6 * 14 * 14 * sizeof(float), cudaMemcpyDeviceToHost);
//     printf("\nMatrice après subsampling (S1_data, profondeur 0) :\n");
//     MatrixPrint(S1_data, 14, 14);

//     // Libération de mémoire
//     cudaFree(d_raw_data);
//     cudaFree(d_C1_data);
//     cudaFree(d_S1_data);
//     cudaFree(d_C1_kernel);
//     // free(raw_data);
//     // free(C1_data);
//     // free(S1_data);
//     // free(C1_kernel);

//     return 0;
// }

// int main() {
//     // Allocation de mémoire sur CPU
//     float *raw_data = (float *)malloc(32 * 32 * sizeof(float));      // Données d'entrée
//     float *C1_data = (float *)malloc(6 * 28 * 28 * sizeof(float));   // Sortie convolution
//     float *S1_data = (float *)malloc(6 * 14 * 14 * sizeof(float));   // Sortie subsampling
//     float *C1_kernel = (float *)malloc(6 * 5 * 5 * sizeof(float));   // Filtres convolution


//     // Initialisation des données
//     initializeRawData(raw_data);
//     initializeC1Data(C1_data);
//     initializeC1Kernel(C1_kernel);
//     initializeS1Data(S1_data);    

//     // Afficher raw_data avant la convolution
//     printf("Matrice d'entrée (raw_data) :\n");
//     MatrixPrint(raw_data, 32, 32);

//     // Allocation de mémoire sur GPU
//     float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;
//     cudaMalloc(&d_raw_data, 32 * 32 * sizeof(float));
//     cudaMalloc(&d_C1_data, 6 * 28 * 28 * sizeof(float));
//     cudaMalloc(&d_S1_data, 6 * 14 * 14 * sizeof(float));
//     cudaMalloc(&d_C1_kernel, 6 * 5 * 5 * sizeof(float));

//     // Copier les données CPU vers GPU
//     cudaMemcpy(d_raw_data, raw_data, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_C1_kernel, C1_kernel, 6 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);

//     // Appliquer la convolution
//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((28 + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (28 + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     for (int d = 0; d < 6; d++) {
//         convolution2D<<<blocksPerGrid, threadsPerBlock>>>(d_raw_data, d_C1_data, d_C1_kernel, d, 32, 28, 5);
//     }

//     cudaDeviceSynchronize();

//     // Copier C1_data sur CPU et afficher
//     cudaMemcpy(C1_data, d_C1_data, 6 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);
//     printf("\nMatrice après convolution (C1_data, profondeur 0) :\n");
//     MatrixPrint(C1_data, 28, 28);

//     // Appliquer le sous-échantillonnage
//     blocksPerGrid = dim3((14 + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                          (14 + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     for (int d = 0; d < 6; d++) {
//         subsampling<<<blocksPerGrid, threadsPerBlock>>>(d_C1_data, d_S1_data, d, 28, 14);
//     }

//     cudaDeviceSynchronize();

//     // Copier S1_data sur CPU et afficher
//     cudaMemcpy(S1_data, d_S1_data, 6 * 14 * 14 * sizeof(float), cudaMemcpyDeviceToHost);
//     printf("\nMatrice après subsampling (S1_data, profondeur 0) :\n");
//     MatrixPrint(S1_data, 14, 14);

//     // Libération de mémoire
//     cudaFree(d_raw_data);
//     cudaFree(d_C1_data);
//     cudaFree(d_S1_data);
//     cudaFree(d_C1_kernel);
//     free(raw_data);
//     free(C1_data);
//     free(S1_data);
//     free(C1_kernel);

//     return 0;
// }

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// PARTIE 3 : A bit of Python //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
// Missing functions ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

__global__ void convolution2D_layer4(float *input, float *output, float *kernel, int depth, int in_width, int out_width, int kernel_size) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < out_width && ty < out_width) {
        int output_idx = depth * out_width * out_width + ty * out_width + tx; // Index pour la sortie
        float sum = 0.0f;

        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int input_x = tx + kx;
                int input_y = ty + ky;
                int input_idx = input_y * in_width + input_x; // Index pour l'entrée
                int kernel_idx = depth * kernel_size * kernel_size + ky * kernel_size + kx; // Index pour le filtre
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }

        // Normaliser la somme pour éviter la saturation
        sum /= (float)(kernel_size * kernel_size); 
        output[output_idx] = tanhf(sum); // Appliquer une fonction tanh pour limiter les valeurs
    }
}


__global__ void fullyConnectedLayer(float *input, float *weights, float *output, int in_size, int out_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < out_size) {
        float sum = 0.0f;
        for (int i = 0; i < in_size; i++) {
            sum += input[i] * weights[idx * in_size + i];
        }
        // Normalisation pour éviter des valeurs extrêmes
        output[idx] = tanhf(sum / in_size);
    }
}


// __global__ void softmax(float *input, float *output, int size) {
//     extern __shared__ float shared[];
//     int idx = threadIdx.x;

//     // Trouver la valeur maximale
//     float max_val = input[0];
//     for (int i = 1; i < size; i++) {
//         max_val = max(max_val, input[i]);
//     }
//     __syncthreads();

//     // Exponentiation
//     shared[idx] = expf(input[idx] - max_val);
//     __syncthreads();

//     // Réduction pour la somme
//     float sum = 0.0f;
//     for (int i = 0; i < size; i++) {
//         sum += shared[i];
//     }
//     __syncthreads();

//     // Normalisation
//     output[idx] = shared[idx] / sum;
// }


/////////////////////////////////////////////////////////////////////////////////////
// functions to import MNIST from binary file //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>

#define WIDTH 28
#define HEIGHT 28

// Fonction pour afficher un pixel avec un fond coloré
void charBckgrndPrint(char *str, int rgb[3]){
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    printf("%s\033[0m", str);
}

// Fonction pour afficher l'image en couleurs
void imgColorPrint(int height, int width, int ***img){
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            charBckgrndPrint("  ", img[row][col]);
        }
        printf("\n");
    }
}// Fonction pour afficher une image normalisée
void displayImage(float *image, int width, int height) {
    printf("Pixels normalisés de la première image :\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.4f ", image[i * width + j]);
        }
        printf("\n");
    }
}
// int main() {
//     int i, j;
//     int ***img;
//     unsigned int magic, nbImg, nbRows, nbCols;
//     unsigned char val;
//     FILE *fptr;

//     // Allocation de mémoire pour l'image
//     img = (int ***)malloc(HEIGHT * sizeof(int **));
//     for(i = 0; i < HEIGHT; i++){
//         img[i] = (int **)malloc(WIDTH * sizeof(int *));
//         for(j = 0; j < WIDTH; j++){
//             img[i][j] = (int *)malloc(3 * sizeof(int));
//         }
//     }

//     // Ouverture du fichier MNIST
//     if((fptr = fopen("train-images.idx3-ubyte", "rb")) == NULL){
//         printf("Impossible d'ouvrir le fichier\n");
//         exit(1);
//     }

//     // Lecture des métadonnées du fichier MNIST
//     fread(&magic, sizeof(int), 1, fptr);
//     fread(&nbImg, sizeof(int), 1, fptr);
//     fread(&nbRows, sizeof(int), 1, fptr);
//     fread(&nbCols, sizeof(int), 1, fptr);

//     // Conversion des entiers en ordre correct (big-endian à little-endian)
//     magic = __builtin_bswap32(magic);
//     nbImg = __builtin_bswap32(nbImg);
//     nbRows = __builtin_bswap32(nbRows);
//     nbCols = __builtin_bswap32(nbCols);

//     printf("Magic Number: %u\n", magic);
//     printf("Nombre d'images: %u\n", nbImg);
//     printf("Dimensions: %u x %u\n", nbRows, nbCols);

//     // Lecture d'une image et application d'une couleur
//     for(i = 0; i < HEIGHT; i++){
//         for(j = 0; j < WIDTH; j++){
//             fread(&val, sizeof(unsigned char), 1, fptr);
//             img[i][j][0] = (int)val; // Rouge
//             img[i][j][1] = (int)val; // Vert
//             img[i][j][2] = (int)val; // Bleu
//         }
//     }

//     // Affichage de l'image
//     imgColorPrint(HEIGHT, WIDTH, img);

//     // Libération de la mémoire
//     for(i = 0; i < HEIGHT; i++){
//         for(j = 0; j < WIDTH; j++){
//             free(img[i][j]);
//         }
//         free(img[i]);
//     }
//     free(img);

//     fclose(fptr);

//     return 0;
// }

/////////////////////////////////////////////////////////////////////////////////////
// Functions to load notebook weigths //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <time.h> // Pour la fonction rand()

// Charger les poids depuis un fichier binaire et afficher 10 poids aléatoires
void loadWeights(const char *filename, float **hostWeights, int numWeights) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Erreur : Impossible d'ouvrir le fichier %s\n", filename);
        exit(1);
    }

    *hostWeights = (float *)malloc(numWeights * sizeof(float));
    if (!*hostWeights) {
        printf("Erreur : Allocation mémoire échouée.\n");
        fclose(file);
        exit(1);
    }

    fread(*hostWeights, sizeof(float), numWeights, file);
    fclose(file);
    printf("Poids chargés depuis le fichier : %s\n", filename);

    // Afficher 10 poids aléatoires
    printf("10 poids aléatoires :\n");
    srand(time(NULL)); // Initialiser le générateur de nombres aléatoires
    for (int i = 0; i < 10; i++) {
        int randomIndex = rand() % numWeights; // Générer un index aléatoire
        printf("Poids[%d] = %f\n", randomIndex, (*hostWeights)[randomIndex]);
    }
}



// int main() {
//     const char *filename = "weights.bin";
//     int numWeights = 185120;  // Nombre total de poids (ajustez en fonction du modèle)
//     float *hostWeights = NULL;

//     // Charger les poids
//     loadWeights(filename, &hostWeights, numWeights);

//     // Allouer les matrices GPU pour chaque couche
//     float *conv1_weights, *conv2_weights, *fc1_weights, *fc2_weights;

//     cudaMalloc((void **)&conv1_weights, 1500 * sizeof(float));  // Taille fictive
//     cudaMalloc((void **)&conv2_weights, 24000 * sizeof(float)); // Taille fictive
//     cudaMalloc((void **)&fc1_weights, 3000 * sizeof(float));    // Taille fictive
//     cudaMalloc((void **)&fc2_weights, 1000 * sizeof(float));    // Taille fictive

//     // Charger les poids dans les matrices GPU
//     int offset = 0;
//     cudaMemcpy(conv1_weights, hostWeights + offset, 1500 * sizeof(float), cudaMemcpyHostToDevice);
//     offset += 1500;

//     cudaMemcpy(conv2_weights, hostWeights + offset, 24000 * sizeof(float), cudaMemcpyHostToDevice);
//     offset += 24000;

//     cudaMemcpy(fc1_weights, hostWeights + offset, 3000 * sizeof(float), cudaMemcpyHostToDevice);
//     offset += 3000;

//     cudaMemcpy(fc2_weights, hostWeights + offset, 1000 * sizeof(float), cudaMemcpyHostToDevice);

//     // Libérer les ressources
//     free(hostWeights);
//     cudaFree(conv1_weights);
//     cudaFree(conv2_weights);
//     cudaFree(fc1_weights);
//     cudaFree(fc2_weights);

//     printf("Poids importés et matrices GPU allouées avec succès.\n");

//     return 0;
// }

/////////////////////////////////////////////////////////////////////////////////////
// Testing the implementation of convolutional neural network inference in C/CUDA //
///////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generateWeights(const char *filename, int numWeights) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Erreur : Impossible de créer le fichier %s\n", filename);
        exit(1);
    }

    float *weights = (float *)malloc(numWeights * sizeof(float));
    if (!weights) {
        printf("Erreur : Allocation mémoire échouée.\n");
        fclose(file);
        exit(1);
    }

    srand(time(NULL)); // Initialisation du générateur de nombres aléatoires
    for (int i = 0; i < numWeights; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2 - 1; // Valeurs entre -1 et 1
    }

    fwrite(weights, sizeof(float), numWeights, file);
    free(weights);
    fclose(file);

    printf("Fichier %s créé avec succès avec %d poids.\n", filename, numWeights);
}

void displayRandomWeights(float *weights, int numWeights) {
    printf("10 poids aléatoires :\n");
    for (int i = 0; i < 10; i++) {
        int randomIndex = rand() % numWeights; // Indice aléatoire
        printf("Poids[%d] = %f\n", randomIndex, weights[randomIndex]);
    }
}

// int main() {
//     const char *filename = "weights.bin";
//     int numWeights = 100000; // Nombre de poids

//     // Générer le fichier de poids
//     generateWeights(filename, numWeights);

//     // Charger les poids depuis le fichier
//     float *weights = NULL;
//     loadWeights(filename, &weights, numWeights);

//     // Afficher 10 poids aléatoires
//     srand(time(NULL)); // Réinitialisation du générateur de nombres aléatoires
//     displayRandomWeights(weights, numWeights);

//     // Libérer la mémoire
//     free(weights);

//     printf("Fin du programme.\n");
//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Constantes
#define WIDTH 28
#define HEIGHT 28
#define NUM_CLASSES 10
#define CONV1_OUT 6
#define CONV1_KERNEL 5
#define FC1_OUT 120
#define FC2_OUT 84

// Fonction pour charger les données MNIST
void loadMNIST(const char *filename, unsigned char **images, int *numImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Erreur : Impossible d'ouvrir le fichier %s\n", filename);
        exit(1);
    }

    unsigned int magic, nbRows, nbCols;
    fread(&magic, sizeof(unsigned int), 1, file);
    fread(numImages, sizeof(unsigned int), 1, file);
    fread(&nbRows, sizeof(unsigned int), 1, file);
    fread(&nbCols, sizeof(unsigned int), 1, file);

    *numImages = __builtin_bswap32(*numImages);
    nbRows = __builtin_bswap32(nbRows);
    nbCols = __builtin_bswap32(nbCols);

    int imageSize = nbRows * nbCols;
    *images = (unsigned char *)malloc((*numImages) * imageSize);
    fread(*images, sizeof(unsigned char), (*numImages) * imageSize, file);
    fclose(file);

    printf("Données MNIST chargées avec succès : %d images de dimensions %dx%d\n", *numImages, nbRows, nbCols);
}
// Fonction CUDA pour la convolution 2D
__global__ void convolution2D(float *input, float *output, float *kernel, int depth, int in_width, int out_width, int kernel_size) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < out_width && ty < out_width) {
        int output_idx = depth * out_width * out_width + ty * out_width + tx;
        float sum = 0.0f;

        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int input_x = tx + kx;
                int input_y = ty + ky;
                int input_idx = input_y * in_width + input_x;
                int kernel_idx = depth * kernel_size * kernel_size + ky * kernel_size + kx;
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }

        sum /= (float)(kernel_size * kernel_size); // Normaliser pour éviter les valeurs extrêmes
        output[output_idx] = tanhf(sum); // Activation tanh
    }
}

// Fonction CUDA pour softmax
__global__ void softmax(float *input, float *output, int size) {
    extern __shared__ float shared[];
    int idx = threadIdx.x;

    // Exponentiation
    shared[idx] = expf(input[idx]);
    __syncthreads();

    // Réduction pour la somme
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += shared[i];
    }
    __syncthreads();

    // Normalisation
    output[idx] = shared[idx] / sum;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Prototypes des fonctions nécessaires
void generateWeights(const char *filename, int numWeights);
void loadWeights(const char *filename, float **hostWeights, int numWeights);
void loadMNIST(const char *filename, unsigned char **images, int *numImages);
void displayImage(float *image, int width, int height);
__global__ void convolution2D(float *input, float *output, float *kernel, int depth, int in_width, int out_width, int kernel_size);
__global__ void softmax(float *input, float *output, int size);

#define WIDTH 28
#define HEIGHT 28
#define NUM_CLASSES 10
#define CONV1_OUT 6
#define CONV1_KERNEL 5
#define FC1_OUT 120
#define FC2_OUT 84

// Fonction principale
int main() {
    const char *imageFile = "train-images.idx3-ubyte";
    const char *weightFile = "weights.bin";

    unsigned char *images;
    float *weights;
    int numImages, numWeights = 185120;

    // Générer les poids si le fichier n'existe pas
    FILE *file = fopen(weightFile, "rb");
    if (!file) {
        printf("Fichier de poids non trouvé. Génération de %d poids...\n", numWeights);
        generateWeights(weightFile, numWeights);
    } else {
        fclose(file);
    }

    // Charger les données MNIST et les poids
    loadMNIST(imageFile, &images, &numImages);
    loadWeights(weightFile, &weights, numWeights);

    // Vérification des poids
    printf("Poids chargés depuis le fichier : %s\n", weightFile);
    printf("Poids chargés (10 aléatoires) :\n");
    srand(time(NULL)); // Initialiser le générateur de nombres aléatoires
    for (int i = 0; i < 10; i++) {
        int randomIndex = rand() % numWeights;
        printf("Poids[%d] = %f\n", randomIndex, weights[randomIndex]);
    }

    // Normaliser la première image
    float *inputImage = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        inputImage[i] = images[i] / 255.0f;
    }
    displayImage(inputImage, WIDTH, HEIGHT);

    // Allouer mémoire GPU
    float *d_input, *d_conv1_out, *d_fc1_out, *d_fc2_out, *d_softmax_out;
    cudaMalloc((void **)&d_input, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&d_conv1_out, CONV1_OUT * 24 * 24 * sizeof(float));
    cudaMalloc((void **)&d_fc1_out, FC1_OUT * sizeof(float));
    cudaMalloc((void **)&d_fc2_out, FC2_OUT * sizeof(float));
    cudaMalloc((void **)&d_softmax_out, NUM_CLASSES * sizeof(float));

    cudaMemcpy(d_input, inputImage, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    // Convolution 1
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((24 + 15) / 16, (24 + 15) / 16);
    convolution2D<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_conv1_out, weights, 1, WIDTH, 24, CONV1_KERNEL);
    cudaDeviceSynchronize();

    float *conv1_out = (float *)malloc(CONV1_OUT * 24 * 24 * sizeof(float));
    cudaMemcpy(conv1_out, d_conv1_out, CONV1_OUT * 24 * 24 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Résultats de la convolution 1 (10 premiers) :\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", conv1_out[i]);
    }
    printf("\n");

    // Fully connected layers et softmax
    softmax<<<1, NUM_CLASSES, NUM_CLASSES * sizeof(float)>>>(d_fc2_out, d_softmax_out, NUM_CLASSES);
    cudaDeviceSynchronize();

    float *softmax_out = (float *)malloc(NUM_CLASSES * sizeof(float));
    cudaMemcpy(softmax_out, d_softmax_out, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sortie softmax :\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Classe %d : %.4f\n", i, softmax_out[i]);
    }

    // Libération mémoire
    free(inputImage);
    free(conv1_out);
    free(softmax_out);
    free(images);
    free(weights);
    cudaFree(d_input);
    cudaFree(d_conv1_out);
    cudaFree(d_fc1_out);
    cudaFree(d_fc2_out);
    cudaFree(d_softmax_out);

    printf("Mémoire libérée avec succès. Fin du programme.\n");
    return 0;
}



