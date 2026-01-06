import pyopencl as cl
import numpy as np
import cv2

def run_visual_test():
    # 1. Setup OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # Kernel de Redução (o mesmo código anterior)
    kernel_code = """
    __kernel void find_min_cost(
        __global const float* costMap,
        __global int* outX, __global int* outY,
        int width, int totalElements,
        __local float* localMin, __local int* localIdx) 
    {
        int lid = get_local_id(0);
        float bestVal = 1e10f;
        int bestIdx = -1;

        for (int i = lid; i < totalElements; i += get_local_size(0)) {
            float val = costMap[i];
            if (val < bestVal) { bestVal = val; bestIdx = i; }
        }

        localMin[lid] = bestVal;
        localIdx[lid] = bestIdx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                if (localMin[lid + stride] < localMin[lid]) {
                    localMin[lid] = localMin[lid + stride];
                    localIdx[lid] = localIdx[lid + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            *outX = localIdx[0] % width;
            *outY = localIdx[0] / width;
        }
    }
    """
    prg = cl.Program(ctx, kernel_code).build()

    # 2. Criar Mapa de Custo Artificial (Simulando uma busca real)
    w, h = 640, 480
    total = w * h
    # Criamos um gradiente para parecer um "vale" de busca
    y, x = np.ogrid[:h, :w]
    target_x, target_y = 400, 250
    # O custo aumenta conforme se afasta do alvo (paraboloide)
    host_map = np.sqrt((x - target_x)**2 + (y - target_y)**2).astype(np.float32)
    # Adicionamos um pouco de ruído para testar a robustez
    host_map += np.random.normal(0, 5, (h, w)).astype(np.float32)

    # 3. Execução na GPU
    mf = cl.mem_flags
    cost_map_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_map)
    res_x_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, 4)
    res_y_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, 4)

    local_size = 256
    prg.find_min_cost(queue, (local_size,), (local_size,),
                      cost_map_gpu, res_x_gpu, res_y_gpu,
                      np.int32(w), np.int32(total),
                      cl.LocalMemory(local_size * 4), cl.LocalMemory(local_size * 4))

    found_x = np.empty(1, dtype=np.int32)
    found_y = np.empty(1, dtype=np.int32)
    cl.enqueue_copy(queue, found_x, res_x_gpu)
    cl.enqueue_copy(queue, found_y, res_y_gpu)

    # 4. Visualização com OpenCV
    # Normalizar o mapa para 0-255 para exibição
    display_map = cv2.normalize(host_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Aplicar um ColorMap para facilitar a visão do "vale"
    # display_map = cv2.applyColorMap(display_map, cv2.COLORMAP_JET)

    # Desenhar um alvo onde o KERNEL encontrou o mínimo
    pos = (int(found_x[0]), int(found_y[0]))
    cv2.drawMarker(display_map, pos, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(display_map, pos, 10, (0, 255, 0), 2)
    
    cv2.putText(display_map, f"Encontrado: {pos}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Teste de Reducao OpenCL (Mapa de Custo)", display_map)
    print(f"Alvo real: ({target_x}, {target_y}) | Detectado: {pos}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visual_test()