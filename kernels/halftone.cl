__kernel void halftone(__global const uchar4 *input, __global uchar4 *output, uint in_width, uint out_width,
                       uint out_height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= out_width || y >= out_height)
        return;

    int idx = y * in_width + x;
    uchar4 pixel = input[idx];

    // Simple threshold-based halftone (adjust pattern as needed)
    float intensity = pixel.x / 255.0f;       // Assuming grayscale input
    uchar value = intensity > 0.5f ? 255 : 0; // Binary output
    output[y * out_width + x] = (uchar4) (value, value, value, pixel.w);
}