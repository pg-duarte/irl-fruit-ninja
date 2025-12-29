import pyopencl as cl
import numpy as np
import cv2

def test_match_template():
    # 1. Setup OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # Kernel NCC que você forneceu
    kernel_code = """
    __kernel void match_template(
        __read_only image2d_t frame,
        __read_only image2d_t templateImg,
        __global float* costMap,
        int2 frameDim, int2 templateDim) 
    {
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x < (frameDim.x - templateDim.x) && y < (frameDim.y - templateDim.y)) {
            float sumF = 0.0f, sumT = 0.0f;
            float sumFF = 0.0f, sumTT = 0.0f, sumFT = 0.0f;
            const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

            for (int j = 0; j < templateDim.y; j++) {
                for (int i = 0; i < templateDim.x; i++) {
                    float f = read_imagef(frame, sampler, (int2)(x + i, y + j)).x;
                    float t = read_imagef(templateImg, sampler, (int2)(i, j)).x;
                    sumF += f; sumT += t;
                    sumFF += f * f; sumTT += t * t; sumFT += f * t;
                }
            }

            float numPixels = (float)(templateDim.x * templateDim.y);
            float numerator = sumFT - (sumF * sumT) / numPixels;
            float denominator = sqrt((sumFF - (sumF * sumF) / numPixels) * (sumTT - (sumT * sumT) / numPixels));
            
            float ncc = (denominator > 0.00001f) ? (numerator / denominator) : 0;
            costMap[y * (frameDim.x - templateDim.x) + x] = 1.0f - ncc; 
        }
    }
    """
    prg = cl.Program(ctx, kernel_code).build()

    # 2. Criar dados de teste (Imagem real ou sintética)
    # Vamos criar um fundo cinza e colocar um "alvo" nele
    frame_h, frame_w = 480, 640
    frame_np = (np.random.rand(frame_h, frame_w) * 0.1).astype(np.float32) # Ruído de fundo
    
    # Criar um template (um quadrado com um padrão específico)
    t_h, t_w = 40, 40
    template_np = np.zeros((t_h, t_w), dtype=np.float32)
    cv2.circle(template_np, (20, 20), 15, 0.8, -1) # Um círculo branco no template
    
    # Inserir o template no frame em uma posição conhecida
    true_x, true_y = 300, 200
    frame_np[true_y:true_y+t_h, true_x:true_x+t_w] += template_np

    # 3. Preparar Buffers GPU
    fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    
    img_gpu = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                       fmt, shape=(frame_w, frame_h), hostbuf=frame_np)
    
    temp_gpu = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                        fmt, shape=(t_w, t_h), hostbuf=template_np)
    
    # Mapa de custo de saída
    out_w, out_h = frame_w - t_w, frame_h - t_h
    cost_map_gpu = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out_w * out_h * 4)

    # 4. Executar
    prg.match_template(queue, (out_w, out_h), None, 
                       img_gpu, temp_gpu, cost_map_gpu, 
                       cl.cltypes.make_int2(frame_w, frame_h), 
                       cl.cltypes.make_int2(t_w, t_h))

    # 5. Recuperar e Visualizar
    host_cost_map = np.empty((out_h, out_w), dtype=np.float32)
    cl.enqueue_copy(queue, host_cost_map, cost_map_gpu)

    # Encontrar o mínimo no mapa (melhor match)
    min_val = np.min(host_cost_map)
    min_loc = np.unravel_index(np.argmin(host_cost_map), host_cost_map.shape)
    # min_loc retorna (y, x)

    
    # Mostrar resultados
    print(f"Alvo real em: ({true_x}, {true_y})")
    print(f"Melhor match encontrado em: ({min_loc[1]}, {min_loc[0]}) com custo {min_val:.4f}")

    # Visualização
    vis_map = cv2.normalize(host_cost_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.circle(vis_map, (int(min_loc[1]), int(min_loc[0])), 30, (0, 255, 0), 4)
    # vis_map = cv2.applyColorMap(255 - vis_map, cv2.COLORMAP_HOT) # Invertemos para o melhor ser "quente"
    
    cv2.imshow("Mapa de Custo (Match)", vis_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_match_template()