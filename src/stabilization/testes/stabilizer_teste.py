import pyopencl as cl
import numpy as np
import cv2

def test_stabilize_visual():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    kernel_code = """
    __kernel void stabilize(
        __read_only image2d_t input, __write_only image2d_t output,
        int2 offset, float zoom) 
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        int2 dims = get_image_dim(input);
        const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

        float normX = (float)x / dims.x;
        float normY = (float)y / dims.y;

        float centerX = (normX - 0.5f) / zoom + 0.5f;
        float centerY = (normY - 0.5f) / zoom + 0.5f;

        float2 readPos = (float2)(
            centerX + (float)offset.x / dims.x,
            centerY + (float)offset.y / dims.y
        );

        float4 color = read_imagef(input, sampler, readPos);
        write_imagef(output, (int2)(x, y), color);
    }
    """
    prg = cl.Program(ctx, kernel_code).build()

    # 1. Criar uma imagem de teste (um padrão de grade com um círculo)
    h, w = 480, 640
    img = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 40): cv2.line(img, (0, i), (w, i), 0.5, 1)
    for i in range(0, w, 40): cv2.line(img, (i, 0), (i, h), 0.5, 1)
    cv2.circle(img, (w//2, h//2), 50, 1.0, -1)
    cv2.putText(img, "TOP-LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 1.0, 2)

    # 2. Buffers
    fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    img_gpu = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                       fmt, shape=(w, h), hostbuf=img)
    out_gpu = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

    # 3. Testar diferentes parâmetros
    # Exemplo: Deslocar 50 pixels para a direita e dar 1.5x de zoom
    offset = cl.cltypes.make_int2(50, 0)
    zoom = np.float32(1.1)

    prg.stabilize(queue, (w, h), None, img_gpu, out_gpu, offset, zoom)

    # 4. Recuperar e mostrar
    result = np.empty((h, w), dtype=np.float32)
    cl.enqueue_copy(queue, result, out_gpu, origin=(0,0), region=(w, h))

    # Mostrar Original vs Estabilizado
    cv2.imshow("Original", img)
    cv2.imshow("Estabilizado (Zoom + Offset)", result)
    print("Pressione qualquer tecla para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_stabilize_visual()