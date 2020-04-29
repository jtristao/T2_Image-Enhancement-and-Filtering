################################################################################
#                          Universidade de São Paulo                           #
#                       B.Sc. Computer Science - 2020/1                        #
#                          SCC0251 - Image Processing                          #
#                                                                              #
#                Assignment 2: Image Enhancement and Filtering                 #
#                                                                              #
#                       Gabriel Kanegae Souza - 10262648                       #
#                   João Vitor dos Santos Tristão - 10262652                   #
################################################################################

import numpy as np
import imageio

# Erro quadratico
def root_square_error(input_img, output_img):
    rse = np.sqrt(np.sum(np.power(output_img - input_img, 2)))

    return rse

# Gaussiana
def gaussian(x, sigma):
    return np.exp((-x**2)/(2*sigma**2))/(2*np.pi*sigma**2)

# Padding de n linhas e colunas com 0
def padding(img, n):
    img = np.pad(img, [(n, n), (n, n)], mode='constant', constant_values=0)

    return img

# Unpadding de n linhas e colunas
def unpadding(img, n):
    img = img[n:img.shape[0]-n, n:img.shape[1]-n]

    return img

# Normaliza a imagem entre 0 e 255
def normalization(img):
    max_val = img.max()
    min_val = img.min()

    return ((img-min_val)*255)/max_val

# Controi o filtro spatial gaussian
def build_filter(n, sigma):
    filter_ = np.zeros((n,n))

    begin = n//2

    for i in range(n):
        for j in range(n):
            val = ((i-begin)**2 + (j-begin)**2)**0.5
            filter_[i][j] = gaussian(val, sigma)

    return filter_


# Aplica o filtro bilateral usando gaussianas
def bilateral_filter(img, n, sigma_s, sigma_r):
    filter_ = build_filter(n, sigma_s)
    
    original_shape = list(img.shape)

    pad = n//2
    img = padding(img, pad)
    
    new_img = np.zeros_like(img)

    for i in range(pad, original_shape[0]+pad):
        for j in range(pad, original_shape[1]+pad):
            # As operacoes aqui acontecem de forma vetorial em torno de img[i][j] (E um pouco chato acompanhar os indices)
            
            # Grid centrada em img[i][j]
            sub_matrix = img[i-pad:i+pad+1, j-pad:j+pad+1]

            gr = gaussian(sub_matrix-img[i][j], sigma_r)
            
            wt = np.multiply(gr, filter_)
            w = np.sum(wt)

            pixel = np.sum(np.multiply(wt, sub_matrix))
            pixel = pixel/w

            new_img[i][j] = pixel

    new_img = unpadding(new_img, pad)
    
    return new_img

# Aplica o filtro de laplace
def laplacian_filter(img, c, kernel_val):
    if kernel_val == 1:
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    elif kernel_val == 2:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


    original_shape = list(img.shape)

    img = padding(img, 1)
    filtered_img = np.zeros_like(img)

    for i in range(1, original_shape[0]+1):
        for j in range(1, original_shape[1]+1):
            sub_matrix = img[i-1:i+2, j-1:j+2]
            filtered_img[i][j] = np.sum(np.multiply(kernel, sub_matrix))

    filtered_img = unpadding(filtered_img, 1)
    img = unpadding(img, 1)

    filtered_img = normalization(filtered_img)
    
    img = filtered_img*c + img

    img = normalization(img)

    return img

# Constroi o vetor gaussian kernel
def gaussian_kernel(dim, sigma):
    kernel = np.zeros(dim)

    if dim%2 == 0:
        begin = dim//2-1
    else:
        begin = dim//2

    for i in range(dim):
        kernel[i] = gaussian(i-begin, sigma)

    return kernel

# Aplica o filtro vignette
def vignette_filter(img, sigma_row, sigma_col):
    dim = list(img.shape)
    
    w_row = gaussian_kernel(dim[0], sigma_row)
    w_row = w_row.reshape((-1, 1))


    w_col = gaussian_kernel(dim[1], sigma_col)

    w = w_row * w_col

    img = img*w
    img = normalization(img)

    return img

filename = str(input()).rstrip()
input_img = imageio.imread(filename)
method = int(input())
save = int(input())

# Converte a imagem de inteiro para float
input_img = np.asarray(input_img, dtype=float)

if method == 1:
    n = int(input())
    sigma_s = float(input())
    sigma_r = float(input())

    output_img = bilateral_filter(input_img, n, sigma_s, sigma_r)
    
elif method == 2:
    c = float(input())
    kernel = int(input())

    output_img = laplacian_filter(input_img, c, kernel)

elif method == 3:
    sigma_row = float(input())
    sigma_col = float(input())

    output_img = vignette_filter(input_img, sigma_row, sigma_col)

else:
    print("Invalid Method")
    exit()

# Imprime o erro quadratico
print("{:.4f}".format(root_square_error(input_img, output_img)))

if save == 1:
    output_img = np.asarray(output_img, dtype="uint8")
    imageio.imwrite("output_img.png", output_img)