import pyopencl as cl
import numpy as np

class GPUImageStabilizer:

    KERNEL_SOURCE = """
        __kernel void match_template_multi_dims(
            __read_only image2d_t frame,
            __read_only image2d_t templateImg,
            __global float* costMap,
            int2 frameDim, 
            int2 templateDim,
            int offsetX, int offsetY,
            float startAngle, float angleStep,
            float startScale, float scaleStep,
            int numScales) // Precisamos saber quantas escalas existem
        {
            int x_local = get_global_id(0);
            int y_local = get_global_id(1);
            
            // O id 2 agora controla Rotação E Escala
            int combined_idx = get_global_id(2);
            int scale_idx = combined_idx % numScales;
            int angle_idx = combined_idx / numScales;

            int x = x_local + offsetX;
            int y = y_local + offsetY;

            float angle = startAngle + (angle_idx * angleStep);
            float scale = startScale + (scale_idx * scaleStep);
            
            float cosA = cos(angle);
            float sinA = sin(angle);
            float2 centerT = (float2)(templateDim.x * 0.5f, templateDim.y * 0.5f);

            // Verificação de limites básica
            if (x < (frameDim.x - templateDim.x) && y < (frameDim.y - templateDim.y)) {
                float sumF = 0.0f, sumT = 0.0f, sumFF = 0.0f, sumTT = 0.0f, sumFT = 0.0f;
                const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

                // Iteramos sobre o tamanho original do template
                for (int j = 0; j < templateDim.y; j++) {
                    for (int i = 0; i < templateDim.x; i++) {
                        // Pixel do frame (fixo)
                        float f = read_imagef(frame, sampler, (int2)(x + i, y + j)).x;

                        // Cálculo do ponto no template com Rotação e ESCALA
                        // p é a distância relativa ao centro
                        float2 p = (float2)(i - centerT.x, j - centerT.y);
                        
                        // Aplicamos a escala inversa para "mapear" o tamanho atual do frame de volta ao template
                        float2 scaledP = p * (1.0f / scale);

                        float2 rotP = (float2)(
                            scaledP.x * cosA - scaledP.y * sinA + centerT.x,
                            scaledP.x * sinA + scaledP.y * cosA + centerT.y
                        );

                        // Lemos o template (o sampler linear ajuda muito aqui no redimensionamento)
                        float t = read_imagef(templateImg, sampler, rotP).x;

                        sumF += f; sumT += t;
                        sumFF += f * f; sumTT += t * t; sumFT += f * t;
                    }
                }

                float numPixels = (float)(templateDim.x * templateDim.y);
                
                // --- Cálculo de NCC e Brilho ---
                float avgF = sumF / numPixels;
                float avgT = sumT / numPixels;
                float brightnessDiff = fabs(avgF - avgT);

                float numerator = sumFT - (sumF * sumT) / numPixels;
                float denomVal = (sumFF - (sumF * sumF) / numPixels) * (sumTT - (sumT * sumT) / numPixels);
                float ncc = (denomVal > 0.0001f) ? (numerator / sqrt(denomVal)) : 0;

                // Custo final
                float cost = (1.0f - ncc) + (brightnessDiff * 0.5f);

                // Mapeamento no costMap (X, Y, Combined_Idx)
                int sliceSize = get_global_size(0) * get_global_size(1);
                costMap[combined_idx * sliceSize + (y_local * get_global_size(0) + x_local)] = cost;
            }
        }

        // 2. NOVO: Kernel de Redução para encontrar o mínimo (O que faltava!)
        __kernel void find_min_cost_3d(
            __global const float* costMap,
            __global int* outX, __global int* outY, __global int* outAngleIdx,
            int width, int areaPerAngle, int totalElements,
            __local float* localMin, __local int* localIdx) 
        {
            int lid = get_local_id(0);
            float bestVal = 1e10f;
            int bestIdx = -1;

            // Varredura global
            for (int i = lid; i < totalElements; i += get_local_size(0)) {
                float val = costMap[i];
                if (val < bestVal) {
                    bestVal = val;
                    bestIdx = i;
                }
            }

            localMin[lid] = bestVal;
            localIdx[lid] = bestIdx;
            barrier(CLK_LOCAL_MEM_FENCE);

            // Redução local (árvore)
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
                int globalIdx = localIdx[0];
                
                // Decompor o índice global em Ângulo, Y e X
                // Estrutura do custo: [Angle][Y][X]
                int angleIdx = globalIdx / areaPerAngle;
                int remains = globalIdx % areaPerAngle;
                
                *outAngleIdx = angleIdx;
                *outY = remains / width;
                *outX = remains % width;
            }
        }

        // 3. Kernel de Estabilização
        __kernel void stabilize(
            __read_only image2d_t input,
            __write_only image2d_t output,
            int2 offset,
            float zoom) 
        {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int2 dims = get_image_dim(input);

            // 1. Sampler com Filtro Linear é essencial para o zoom não ficar "pixelizado"
            const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

            // 2. Calcular coordenadas normalizadas (0.0 a 1.0) para o centro
            float normX = (float)x / dims.x;
            float normY = (float)y / dims.y;

            // 3. Aplicar o Zoom e o Deslocamento (offset)
            // O ajuste (0.5f * (1.0f - 1.0f/zoom)) garante que o zoom seja a partir do centro
            float centerX = (normX - 0.5f) / zoom + 0.5f;
            float centerY = (normY - 0.5f) / zoom + 0.5f;

            // 4. Adicionar o deslocamento da estabilização (também normalizado)
            float2 readPos = (float2)(
                centerX + (float)offset.x / dims.x,
                centerY + (float)offset.y / dims.y
            );

            float4 color = read_imagef(input, sampler, readPos);
            write_imagef(output, (int2)(x, y), color);
        }
    """

    def __init__(self, template_img, initial_x, initial_y, frame_shape, faceMargin=80, numAngles=9):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        
        self.x0, self.y0 = initial_x, initial_y
        self.h_f, self.w_f = frame_shape
        self.h_t, self.w_t = template_img.shape
        self.margin = faceMargin
        self.w_match = faceMargin * 2
        self.h_match = faceMargin * 2
        self.last_angle = 0.0  # Ângulo inicial
        self.num_angles = numAngles

        #Fator de suavização para os deslocamentos
        self.smooth_dx, self.smooth_dy = 0, 0
        # Tamanho do local size para o kernel de redução
        self.local_size = 256

        # Tenta compilar e mostra erro detalhado se falhar
        try:
            self.prg = cl.Program(self.ctx, self.KERNEL_SOURCE).build()
        except cl.RuntimeError as e:
            print(self.prg.get_build_info(self.ctx.devices[0], cl.program_build_info.LOG))
            raise e

        self.fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        
        self.template_gpu = cl.Image(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                                    self.fmt, shape=(self.w_t, self.h_t), 
                                    hostbuf=template_img.astype(np.float32))
        
        self.output_gpu = cl.Image(self.ctx, self.mf.WRITE_ONLY, self.fmt, shape=(self.w_f, self.h_f))
        self.cost_map_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, self.w_match * self.h_match * self.num_angles *4)
        
        self.res_x_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, 4)
        self.res_y_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, 4)
        self.res_angle_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, 4) # Buffer para o índice do ângulo

    def process(self, frame_np, zoom=1.2):
        # 0. Upload do frame
        frame_gpu = cl.Image(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                            self.fmt, shape=(self.w_f, self.h_f), 
                            hostbuf=frame_np.astype(np.float32))

        # --- LÓGICA DA JANELA DE BUSCA ---
        # Centralizamos a busca onde o objeto estava (x0 + deslocamento atual)
        curr_x = self.x0 + int(round(self.smooth_dx))
        curr_y = self.y0 + int(round(self.smooth_dy))

        # Calculamos o canto superior esquerdo da janela de busca
        # self.margin deve ser definido no __init__ (ex: 50)
        start_x = max(0, min(curr_x - self.margin, self.w_f - self.w_t - self.w_match))
        start_y = max(0, min(curr_y - self.margin, self.h_f - self.h_t - self.h_match))

        # 1. Matching Rotacionado
        angle_step = np.float32(np.radians(5)) 
        start_angle = np.float32(self.last_angle - (self.num_angles // 2) * angle_step)
        
        f_dim = cl.cltypes.make_int2(self.w_f, self.h_f)
        t_dim = cl.cltypes.make_int2(self.w_t, self.h_t)
        global_size = (self.w_match, self.h_match, self.num_angles)



        # __read_only image2d_t frame,
        #     __read_only image2d_t templateImg,
        #     __global float* costMap,
        #     int2 frameDim, 
        #     int2 templateDim,
        #     int offsetX, int offsetY,
        #     float startAngle, float angleStep,
        #     float startScale, float scaleStep,
        #     int numScales) // Precisamos saber quantas escalas existem
        # {
        self.prg.match_template_multi_dims(
            self.queue, global_size, None,
            frame_gpu, self.template_gpu, self.cost_map_gpu,
            f_dim, t_dim, np.int32(start_x), np.int32(start_y),
            start_angle, angle_step, np.float32(1), np.float32(0.2), np.int32(2)
        )

        # 2. Redução 3D (CORRIGIDO: nome da função e argumentos)
        area_per_angle = np.int32(self.w_match * self.h_match)
        total_elements = np.int32(area_per_angle * self.num_angles)

        self.prg.find_min_cost_3d(
            self.queue, (self.local_size,), (self.local_size,),
            self.cost_map_gpu, 
            self.res_x_gpu, self.res_y_gpu, self.res_angle_gpu,
            np.int32(self.w_match), area_per_angle, total_elements,
            cl.LocalMemory(self.local_size * 4), cl.LocalMemory(self.local_size * 4)
        )

        # 3. Recuperar coordenadas e ÂNGULO
        x_local = np.empty(1, dtype=np.int32)
        y_local = np.empty(1, dtype=np.int32)
        ang_idx = np.empty(1, dtype=np.int32)
        
        cl.enqueue_copy(self.queue, x_local, self.res_x_gpu)
        cl.enqueue_copy(self.queue, y_local, self.res_y_gpu)
        cl.enqueue_copy(self.queue, ang_idx, self.res_angle_gpu)

        # Atualizar ângulo para o próximo frame
        self.last_angle = start_angle + (ang_idx[0] * angle_step)

        # Coordenada global = início da janela + posição encontrada dentro dela
        x1_global = start_x + x_local[0]
        y1_global = start_y + y_local[0]

        # Cálculo do deslocamento para estabilização
        dx, dy = np.int32(x1_global - self.x0), np.int32(y1_global - self.y0)
        
        # Filtro de suavização
        alpha = 0.2
        self.smooth_dx = alpha * dx + (1 - alpha) * self.smooth_dx
        self.smooth_dy = alpha * dy + (1 - alpha) * self.smooth_dy

        offset_arg = cl.cltypes.make_int2(
            np.int32(round(self.smooth_dx)), 
            np.int32(round(self.smooth_dy))
        )

        # 4. Estabilização (usa o frame inteiro)
        self.prg.stabilize(self.queue, (self.w_f, self.h_f), None, 
                        frame_gpu, self.output_gpu, offset_arg, np.float32(zoom))

        # 5. Resultado
        result = np.empty((self.h_f, self.w_f), dtype=np.float32)
        cl.enqueue_copy(self.queue, result, self.output_gpu, origin=(0,0), region=(self.w_f, self.h_f))

        self.last_match_x = x1_global
        self.last_match_y = y1_global
        
        return result, x1_global, y1_global, np.degrees(self.last_angle)
    
    def get_cost_map_visual(self):
        # 1. Criar um buffer no Python para receber os dados
        # O tamanho é (h_match, w_match)
        host_cost_map = np.empty((self.h_match, self.w_match), dtype=np.float32)
        
        # 2. Copiar da GPU para a RAM
        cl.enqueue_copy(self.queue, host_cost_map, self.cost_map_gpu)
        
        # 3. Normalizar: O menor valor (melhor match) vira 0, o maior vira 255
        min_val = np.min(host_cost_map)
        max_val = np.max(host_cost_map)
        
        # Evitar divisão por zero e converter para 8 bits
        if max_val - min_val > 0:
            map_rescaled = (host_cost_map - min_val) / (max_val - min_val) * 255
        else:
            map_rescaled = host_cost_map
            
        return map_rescaled.astype(np.uint8)