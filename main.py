import math

from PIL import Image
from math import log2
import os
import time


def read_bmp_image(image_path):
    try:
        img = Image.open(image_path)

        pixel_val = []
        width, height = img.size
        for y in range(height):
            row = []
            for x in range(width):
                row.append(img.getpixel((x, y)))
            pixel_val.append(row)

        return img, pixel_val, width, height, img.mode

    except FileNotFoundError:
        print("File not found!")
    except Exception as e:
        print(f"An error occurred: {e}")


def predict(P, X, Y):
    E = []

    for x in range(Y):
        for y in range(X):
            value_to_append = P.getpixel((x, y))

            if x == 0 and y == 0:
                pass
            elif y == 0:
                value_to_append = P.getpixel((x - 1, 0)) - P.getpixel((x, 0))
            elif x == 0:
                value_to_append = P.getpixel((0, y - 1)) - P.getpixel((0, y))
            else:
                current_val = P.getpixel((x, y))
                max_val = max(P.getpixel((x - 1, y)), P.getpixel((x, y - 1)))
                min_val = min(P.getpixel((x - 1, y)), P.getpixel((x, y - 1)))

                if P.getpixel((x - 1, y - 1)) >= max_val:
                    value_to_append = min_val - current_val
                elif P.getpixel((x - 1, y - 1)) <= min_val:
                    value_to_append = max_val - current_val
                else:
                    value_to_append = (
                            P.getpixel((x - 1, y))
                            + P.getpixel((x, y - 1))
                            - P.getpixel((x - 1, y - 1))
                            - current_val
                    )
            E.append(value_to_append)

    return E


def setHeader(X, min_val, max_val, n):
    header = []

    # Image height (12 bits)
    header.append(bin(n)[2:].zfill(12))

    # First element from C (8 bits)
    header.append(bin(min_val)[2:].zfill(8))

    # Last element from C (32 bits)
    header.append(bin(max_val)[2:].zfill(32))

    # Number of all elements (24 bits)
    header.append(bin(X * n)[2:].zfill(24))

    return header


def IC(B, C, L, H):
    if H - L > 1:
        if C[L] != C[H]:
            m = math.floor(0.5 * (L + H))
            g = math.ceil(log2(C[H] - C[L] + 1))
            B = encode(B, g, C[m] - C[L])

            if L < m:
                IC(B, C, L, m)

            if m < H:
                IC(B, C, m, H)
    return B


def encode(B, g, m):
    # Truncate 'm' to fit within 'g' bits
    max_val = (1 << int(g)) - 1
    if m > max_val:
        m = max_val

    # Encode 'm' as a truncated binary with 'g' bits
    truncated_binary = format(m, f'0{int(g)}b')

    # Ensure that the encoded binary matches the size of 'g'
    if len(truncated_binary) > int(g):
        truncated_binary = truncated_binary[-int(g):]  # Take the least significant 'g' bits
    elif len(truncated_binary) < int(g):
        truncated_binary = truncated_binary.zfill(int(g))  # Zero-pad to make it 'g' bits

    B.append(truncated_binary)
    return B


def compress(P, X, Y):
    predicted_val = predict(P, X, Y)
    N = [predicted_val[0]]

    for i in range(1, X*Y):
        if (predicted_val[i] == 0):
            N.append(0)
        elif (predicted_val[i] > 0):
            N.append(2 * predicted_val[i])
        else:
            N.append(2 * abs(predicted_val[i]) - 1)

    C = [N[0]]
    for i in range(1, X*Y):
        C.append(N[i] + C[i - 1])

    n = X*Y
    B = setHeader(X, C[0], C[n-1], n)
    Bic = IC(B, C, 0, n-1)
    return predicted_val, N, C, B, Bic


def decodeheader(B):
    n = int(B[0], 2)
    min_val = int(B[1], 2)
    max_val = int(B[2], 2)
    X = int(B[3], 2) // n

    return X, min_val, max_val, n


def initializeC(n, first_el, last_el):
    return [first_el] + [0] * (n - 2) + [last_el]


def getBits(B, i, g):
    bits = ""
    for j in range(i, len(B)):
        bits += B[j]

        # Increment 'i' to move to the next segment of 'B'
        i += 1
        if len(bits) == g:
            break

    return bits, i  # Return the extracted bits and the updated index 'i'

def decode(bits_list):
    decoded_values = []
    for bits in bits_list:
        decoded_values.append(int(bits, 2))
    return decoded_values


def deIC(B, C, L, H, i=4):
    if H - L > 1:
        if C[L] != C[H]:
            m = math.floor(0.5 * (L + H))
            g = math.ceil(log2(C[H] - C[L] + 1))
            bits, i = getBits(B, i, g)
            decoded_values = decode([bits])
            C[m] = decoded_values[0] + C[L]

            if L < m:
                C, i = deIC(B, C, L, m, i)

            if m < H:
                C, i = deIC(B, C, m, H, i)
        else:
            for j in range(L + 1, H):
                C[j] = C[L]

    return C, i


def inversePredict(E, X, Y):
    P = []

    for x in range(Y):
        for y in range(X):
            index = x * X + y

            if x == 0 and y == 0:
                P.append(E[index])
            elif y == 0:
                x1 = P[(x - 1) * X] - E[index]
                P.append(x1)
            elif x == 0:
                y1 = P[y - 1] - E[index]
                P.append(y1)
            else:
                max_val = max(P[(x - 1) * X + y], P[x * X + y - 1])
                min_val = min(P[(x - 1) * X + y], P[x * X + y - 1])

                if P[(x - 1) * X + y - 1] >= max_val:
                    P.append(min_val - E[index])
                elif P[(x - 1) * X + y - 1] <= min_val:
                    P.append(max_val - E[index])
                else:
                    tmp = (P[(x - 1) * X + y] + P[x * X + y - 1] - P[(x - 1) * X + y - 1])
                    P.append(tmp - E[index])

    return P


def create_image_from_P(P, X, Y):
    img = Image.new('RGB', (Y, X))

    for x in range(Y):
        for y in range(X):
            index = x * X + y
            img.putpixel((x, y), (P[index], P[index], P[index]))

    return img

def decompress(B):
    X, first_el, last_el, n = decodeheader(B)

    # print("X:", X)
    # print("First element:", first_el)
    # print("Last element:", last_el)
    # print("n:", n)

    Y = int(n / X)
    # print("Y:", Y)
    C = initializeC(n, first_el, last_el)
    # print("C:", C)
    C = deIC(B, C, 0, n-1)
    C = C[0]
    # print("C:", C)
    dN = [0] * n
    dN[0] = C[0]
    for i in range(1, n):
        dN[i] = C[i] - C[i - 1]
    # print("dN:", dN)
    dE = [0] * n
    dE[0] = dN[0]

    for i in range(1, n):
        if (dN[i] % 2 == 0):
            dE[i] = int(dN[i] / 2)
        else:
            dE[i] = - int((dN[i] + 1) / 2)
    # print("dE:", dE)
    dP = inversePredict(dE, X, Y)
    dImg = create_image_from_P(dP, X, Y)
    # print("dP:", dP)
    return dImg


def save_as_bmp(dec_img, file_path, org_mode):
    try:
        if org_mode == 'L':
            dec_img = dec_img.convert('L')  # Convert to 8-bit grayscale
        elif org_mode == 'RGB':
            dec_img = dec_img.convert('RGB')  # Convert to 24-bit RGB
        dec_img.save(file_path)

        print(f"Image saved successfully at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    my_img = Image.open(file_path)
    my_img.show()


def compressImages(image_path):
    # Measure original size
    original_size = os.path.getsize(image_path)
    original_size_mb = original_size / (1024 * 1024)  # Convert bytes to MB

    # Load image
    img = Image.open(image_path)
    Y, X = img.size

    # Measure compression time
    start_compress = time.time()
    predicted_values, N, C, B, Bic = compress(img, X, Y)
    end_compress = time.time()
    compression_time = end_compress - start_compress

    # Measure compressed size
    compressed_size = len(''.join(B)) // 8  # Calculate size in bytes
    compressed_size_mb = compressed_size / (1024 * 1024)  # Convert bytes to MB

    # Measure decompression time
    start_decompress = time.time()
    decImg = decompress(B)
    end_decompress = time.time()
    decompression_time = end_decompress - start_decompress

    # Calculate compression ratio
    compression_ratio = original_size / compressed_size

    # Output the measurements for each image
    print(f"Image: {image_path}")
    print(f"Original Size: {original_size_mb:.2f} MB")
    print(f"Compressed Size: {compressed_size_mb:.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Compression Time: {compression_time:.2f} seconds")
    print(f"Decompression Time: {decompression_time:.2f} seconds")
    print("\n")


if __name__ == "__main__":
    image_path = "slike BMP/Lena.bmp"

    slika, pixel_values, Y, X, original_mode = read_bmp_image(image_path)
    # print("Pixel values:", pixel_values)

    # Klicanje funkcije predict z vašimi podatki
    predicted_values, N, C, B, Bic = compress(slika, X, Y)

    # Izpis rezultatov
    # print("Predicted Values:", predicted_values)
    # print("N Values:", N)
    # print("C Values:", C)
    # print("Header:", B)
    # print("BIC:", Bic)

    decImg = decompress(Bic)
    dec_img_path = "decompressed.bmp"
    save_as_bmp(decImg, dec_img_path, original_mode)

    image_paths = [
        "slike BMP/Lena.bmp",
        "slike BMP/Barbara.bmp",
        "slike BMP/Earth.bmp",
        "slike BMP/Fruits.bmp",
        "slike BMP/Bark.bmp",
        "slike BMP/Mosaic.bmp",
        "slike BMP/Gold.bmp",
        "slike BMP/Bicycle.bmp",
        "slike BMP/Sun.bmp",
        "slike BMP/Zelda.bmp",
    ]

    # compressImages(image_path)
