"""Narrow fixes for pinned DORY output on the pinned AI-deck FreeRTOS SDK."""
def patch_bias(target):
    # DORY serializes 32-bit bias (four bytes/channel); legacy pulp-nn matmul
    # consumes one byte/channel. Preserve the byte-pointer API, fix its stride.
    path = target / 'src/pulp_nn_matmul.c'
    source = path.read_text()
    assert source.count('((int) (*bias++))') == 5
    source = source.replace('((int) (*bias++))', '(*(const int32_t *)bias); bias += sizeof(int32_t)')
    path.write_text(source)
    path = target / 'src/pulp_nn_conv_Ho_parallel.c'
    source = path.read_text()
    assert source.count('((int)(bias[i]))') == 1
    source = source.replace('((int)(bias[i]))', '((const int32_t *)bias)[i]')
    path.write_text(source)
