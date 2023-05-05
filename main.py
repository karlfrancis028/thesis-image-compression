import numpy as np
import pywt
from PIL import Image
import heapq
from bitarray import bitarray
import time
import os
from tqdm import tqdm

# # Open image and convert to grayscale
# converted_image = Image.open("image.jpg").convert("L")
#
# # Convert the image to a numpy array
# image_array = np.asarray(converted_image)
#
# # Perform DWT on the image array
# coeffs = pywt.dwt2(image_array, 'haar')
# cA, (cH, cV, cD) = coeffs
#
# def canonical_huffman(coeffs):
#     if len(coeffs) == 0:
#         return None
#     # Create a frequency table for the data
#     frequency_table = {}
#     for arr in coeffs[0]:
#         if arr.shape[0] > 0:
#             for value in arr.flat:
#                 if value in frequency_table:
#                     frequency_table[value] += 1
#                 else:
#                     frequency_table[value] = 1
#     for arr in coeffs[1]:
#         if arr.shape[0] > 0:
#             for value in arr.flat:
#                 if value in frequency_table:
#                     frequency_table[value] += 1
#                 else:
#                     frequency_table[value] = 1
#     # Create a heap from the frequency table
#     heap = [[weight, [symbol, ""]] for symbol, weight in frequency_table.items()]
#     heapq.heapify(heap)
#
#     # Build the Huffman tree
#     while len(heap) > 1:
#         lo = heapq.heappop(heap)
#         hi = heapq.heappop(heap)
#         for pair in lo[1:]:
#             pair[1] = "0" + pair[1]
#         for pair in hi[1:]:
#             pair[1] = "1" + pair[1]
#         heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
#
#     # Create the code table from the Huffman tree
#     code_table = {}
#     for symbol, code in sorted(heap[0][1:], key=lambda p: (len(p[1]), p)):
#         code_table[symbol] = code
#
#     return code_table
#
# # Perform Canonical Huffman on DWT Coefficients
# huffman_encoded_coeffs = canonical_huffman(coeffs)
#
# # Compress the data using Huffman encoded coefficients
# compressed_data = bitarray()
# for arr in coeffs:
#     for value in np.nditer(arr):
#         value = value.item()
#         compressed_data.extend(bitarray(huffman_encoded_coeffs[value]))
#
# print(compressed_data)
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
    compression_ratio = compressed_size / original_size
    print("Compression ratio: {:.2f}".format(compression_ratio))

def decompress_image(compressed_image, code_table, shape):
    print("Decompressing binary now. Please wait...")

    #Start timer
    start_time = time.time()

    # Read the compressed data from file
    with open(compressed_image, "rb") as f:
        compressed_data = bitarray()
        compressed_data.fromfile(f)

    # Decode the compressed data using the Huffman code table
    decoded_data = []
    symbol = ""
    for bit in tqdm(compressed_data, desc='Decompressing image', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        symbol += str(bit)
        if symbol in code_table:
            decoded_data.append(code_table.index(symbol))
            symbol = ""

    # Perform IDWT on the decoded data
    cA, cHVD = np.split(np.array(decoded_data), [shape[0] * shape[1]])
    coeffs = (cA, (cHVD[0:shape[0]*shape[1]//4], cHVD[shape[0]*shape[1]//4:shape[0]*shape[1]//2], cHVD[shape[0]*shape[1]//2:]))
    image_array = pywt.idwt2(coeffs, 'haar')

    # Convert the image array to a PIL image and save it
    image = Image.fromarray(np.uint8(image_array))
    image.save("decompressed_image.jpg")

    #End timer
    end_time = time.time()
    print("Image Decompression Complete!")
    print("Time taken to decompress the image: {:.2f} seconds".format(end_time - start_time))

compress_image("image.jpg")
calculate_compression_ratio("image.jpg", "compressed_image.bin")
huffman_code_table = canonical_huffman(coeffs)
coeffs_shape = (coeffs[0].shape[0], coeffs[0].shape[1])
decompress_image("compressed_image.bin", huffman_code_table, coeffs_shape)
