__kernel void crop(__global const uchar4 *input, __global uchar4 *output, uint in_width, uint out_width,
                   uint out_height, uint start_x, uint start_y) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= out_width || y >= out_height)
        return;

    // Calculate input index
    int in_x = x + start_x;
    int in_y = y + start_y;
    int in_idx = in_y * in_width + in_x;
    int out_idx = y * out_width + x;

    // Copy pixel
    output[out_idx] = input[in_idx];
}