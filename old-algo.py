from PIL import Image
import pywt
import heapq
import collections
import numpy as np
import struct

# Read image file
img = Image.open('image.jpg')

# Convert to grayscale
gray_img = img.convert('L')

# Apply linear prediction
width, height = gray_img.size
prediction_errors = []
for y in range(height):
    for x in range(width):
        if x == 0 or y == 0:
            prediction_error = gray_img.getpixel((x, y))
        else:
            prediction = (gray_img.getpixel((x-1, y)) + gray_img.getpixel((x, y-1))) // 2
            prediction_error = gray_img.getpixel((x, y)) - prediction
        prediction_errors.append(prediction_error)

# Apply integer wavelet transform
coeffs = pywt.wavedec2(np.array(prediction_errors).reshape((height, width)), 'haar', level=1, mode='symmetric', axes=(0, 1))

# Discard redundant coefficients
coeffs_arr = []
for i, arr in enumerate(coeffs):
    if i == 0:
        coeffs_arr.append(arr)
    else:
        coeffs_arr.append(np.sign(arr) * np.floor(np.abs(arr)))

# Build Huffman tree
freq = collections.Counter(coeffs_arr[-1].flatten())
heap = [[weight, [symbol, '']] for symbol, weight in freq.items()]
heapq.heapify(heap)
while len(heap) > 1:
    lo = heapq.heappop(heap)
    hi = heapq.heappop(heap)
    for pair in lo[1:]:
        pair[1] = '0' + pair[1]
    for pair in hi[1:]:
        pair[1] = '1' + pair[1]
    heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

# Generate Huffman codes
huff_codes = dict(heapq.heappop(heap)[1:])

# Encode and save compressed file
with open('compressed.bin', 'wb') as f:
    bitstream = ''
    for i, coeff in enumerate(coeffs_arr[-1].flatten()):
        bitstream += huff_codes[coeff]
        # Write the bitstream to the file in 8-bit chunks
        while len(bitstream) >= 8:
            byte = bitstream[:8]
            bitstream = bitstream[8:]
            f.write(struct.pack('B', int(byte, 2)))
    # If there are any remaining bits in the bitstream, pad with zeros and write to the file
    if len(bitstream) > 0:
        byte = bitstream + '0' * (8 - len(bitstream))
        f.write(struct.pack('B', int(byte, 2))))
