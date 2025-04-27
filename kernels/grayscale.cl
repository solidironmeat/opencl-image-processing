__kernel void grayscale(__global const uchar4 *input, __global uchar4 *output, uint in_width, uint out_width,
                        uint out_height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= out_width || y >= out_height)
        return;

    int idx = y * in_width + x;
    uchar4 pixel = input[idx];

    // Luminance: 0.299R + 0.587G + 0.114B
    uchar gray = (uchar) (0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z + 0.5f);
    output[y * out_width + x] = (uchar4) (gray, gray, gray, pixel.w);
}