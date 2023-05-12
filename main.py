import numpy as np
import pywt
from PIL import Image
import heapq
from bitarray import bitarray
import time
import os
from tqdm import tqdm

#Global Declaration
coeffs = None

def canonical_huffman(coeffs):
    if len(coeffs) == 0:
        return None
    # Create a frequency table for the data
    frequency_table = {}
    for arr in coeffs[0]:
        if arr.shape[0] > 0:
            for value in arr.flat:
                if value in frequency_table:
                    frequency_table[value] += 1
                else:
                    frequency_table[value] = 1
    for arr in coeffs[1]:
        if arr.shape[0] > 0:
            for value in arr.flat:
                if value in frequency_table:
                    frequency_table[value] += 1
                else:
                    frequency_table[value] = 1
    # Create a heap from the frequency table
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency_table.items()]
    heapq.heapify(heap)

    # Build the Huffman tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Create the code table from the Huffman tree
    code_table = {}
    for symbol, code in sorted(heap[0][1:], key=lambda p: (len(p[1]), p)):
        code_table[symbol] = code

    return code_table

def compress_image(image_file):
    print("Compressing image. Please wait...")

    #Start timer
    start_time = time.time()

    global coeffs

    # Open image and convert to grayscale
    converted_image = Image.open(image_file).convert("L")

    # Convert the image to a numpy array
    image_array = np.asarray(converted_image)

    # Perform DWT on the image array
    coeffs = pywt.dwt2(image_array, 'haar')

    # Perform Canonical Huffman on DWT Coefficients
    huffman_code_table = canonical_huffman(coeffs)

    # Compress the data using Huffman encoded coefficients
    compressed_data = bitarray()
    cA, cHVD = coeffs
    for value in tqdm(cA.flat, desc="Compressing cA"):
        compressed_data.extend(bitarray(huffman_code_table[value]))
    for value in tqdm(cHVD[0].flat, desc="Compressing cH"):
        compressed_data.extend(bitarray(huffman_code_table[value]))
    for value in tqdm(cHVD[1].flat, desc="Compressing cV"):
        compressed_data.extend(bitarray(huffman_code_table[value]))
    for value in tqdm(cHVD[2].flat, desc="Compressing cD"):
        compressed_data.extend(bitarray(huffman_code_table[value]))

    # Write the compressed data to a file
    with open("compressed_image.bin", "wb") as f:
        compressed_data.tofile(f)

    #End timer
    end_time = time.time()
    print("Image Compression Complete!")
    print("Time taken to compress the image: {:.2f} seconds".format(end_time - start_time))


def calculate_compression_ratio(original_image, compressed_image):
    original_size = os.path.getsize(original_image)
    compressed_size = os.path.getsize(compressed_image)
    compression_ratio = original_size / compressed_size
    print("Compression ratio: {:.2f}:1".format(compression_ratio))

def decompress_image(image_file):
    print("Decompressing image. Please wait...")

    # Start timer
    start_time = time.time()

    global coeffs

    # Read the compressed data from the file
    with open(image_file, "rb") as f:
        compressed_data = bitarray()
        compressed_data.fromfile(f)
    
    # Perform Canonical Huffman on DWT Coefficients
    huffman_code_table = canonical_huffman(coeffs)

    # Invert the Huffman code table
    inverse_code_table = {code: symbol for symbol, code in huffman_code_table.items()}

    # Decompress the data using Huffman encoded coefficients
    decompressed_data = []
    code = ""
    for bit in tqdm(compressed_data, desc="Decompressing"):
        code += str(bit)
        if code in inverse_code_table:
            symbol = inverse_code_table[code]
            decompressed_data.append(symbol)
            code = ""

    # Convert the decompressed data back to a numpy array
    cA_size = coeffs[0].shape
    cH_size = coeffs[1][0].shape
    cV_size = coeffs[1][1].shape
    cD_size = coeffs[1][2].shape
    cA = np.reshape(decompressed_data[:cA_size[0] * cA_size[1]], cA_size)
    cH = np.reshape(decompressed_data[cA_size[0] * cA_size[1]:cA_size[0] * cA_size[1] + cH_size[0] * cH_size[1]], cH_size)
    cV = np.reshape(decompressed_data[cA_size[0] * cA_size[1] + cH_size[0] * cH_size[1]:cA_size[0] * cA_size[1] + cH_size[0] * cH_size[1] + cV_size[0] * cV_size[1]], cV_size)
    cD = np.reshape(decompressed_data[cA_size[0] * cA_size[1] + cH_size[0] * cH_size[1] + cV_size[0] * cV_size[1]:], cD_size)

    # Perform IDWT on the decompressed data
    decompressed_image_array = pywt.idwt2((cA, (cH, cV, cD)), 'haar')

    # Convert the numpy array to a PIL Image and save to file
    decompressed_image = Image.fromarray(decompressed_image_array.astype(np.uint8))
    decompressed_image.save("decompressed_image.jpg")

    print("Image Decompression Complete. Check the folder for a file named decompressed_image.jpg")

compress_image("image.jpg")
calculate_compression_ratio("image.jpg", "compressed_image.bin")
decompress_image("compressed_image.bin")