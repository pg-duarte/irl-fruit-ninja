import cv2
import pyopencl as cl
import numpy as np

class GPUImageStabilizer:
    # Kernel permanece o mesmo que você postou, mas note que o 'stabilize' 
    # agora vai lidar com float4 (RGBA) automaticamente por causa do formato da imagem.
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
            int numScales) 
        {
            int x_local = get_global_id(0);
            int y_local = get_global_id(1);
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

            if (x < (frameDim.x - templateDim.x) && y < (frameDim.y - templateDim.y)) {
                float sumF = 0.0f, sumT = 0.0f, sumFF = 0.0f, sumTT = 0.0f, sumFT = 0.0f;
                const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

                for (int j = 0; j < templateDim.y; j++) {
                    for (int i = 0; i < templateDim.x; i++) {
                        float f = read_imagef(frame, sampler, (int2)(x + i, y + j)).x;
                        float2 p = (float2)(i - centerT.x, j - centerT.y);
                        float2 scaledP = p * (1.0f / scale);
                        float2 rotP = (float2)(
                            scaledP.x * cosA - scaledP.y * sinA + centerT.x,
                            scaledP.x * sinA + scaledP.y * cosA + centerT.y
                        );
                        float t = read_imagef(templateImg, sampler, rotP).x;
                        sumF += f; sumT += t;
                        sumFF += f * f; sumTT += t * t; sumFT += f * t;
                    }
                }
                float numPixels = (float)(templateDim.x * templateDim.y);
                float avgF = sumF / numPixels;
                float avgT = sumT / numPixels;
                float ncc = (sumFT - (sumF * sumT) / numPixels) / 
                            sqrt(((sumFF - (sumF * sumF) / numPixels) * (sumTT - (sumT * sumT) / numPixels)) + 0.0001f);
                float cost = (1.0f - ncc) + (fabs(avgF - avgT) * 0.5f);
                int sliceSize = get_global_size(0) * get_global_size(1);
                costMap[combined_idx * sliceSize + (y_local * get_global_size(0) + x_local)] = cost;
            }
        }

        __kernel void find_min_cost_3d(
            __global const float* costMap,
            __global int* outX, __global int* outY, __global int* outAngleIdx,
            int width, int areaPerAngle, int totalElements,
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
                if (lid < stride && localMin[lid + stride] < localMin[lid]) {
                    localMin[lid] = localMin[lid + stride];
                    localIdx[lid] = localIdx[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                int gIdx = localIdx[0];
                *outAngleIdx = gIdx / areaPerAngle;
                int remains = gIdx % areaPerAngle;
                *outY = remains / width; *outX = remains % width;
            }
        }

        __kernel void stabilize(
            __read_only image2d_t input,
            __write_only image2d_t output,
            int2 offset, float zoom) 
        {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int2 dims = get_image_dim(input);
            const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
            
            float2 readPos = (float2)(
                ((((float)x / dims.x) - 0.5f) / zoom + 0.5f) + (float)offset.x / dims.x,
                ((((float)y / dims.y) - 0.5f) / zoom + 0.5f) + (float)offset.y / dims.y
            );

            write_imagef(output, (int2)(x, y), read_imagef(input, sampler, readPos));
        }
    """

    def __init__(self, template_img_gray, initial_x, initial_y, frame_shape, faceMargin=80, numAngles=9):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        
        # Parâmetros
        self.x0, self.y0 = initial_x, initial_y
        self.h_f, self.w_f = frame_shape
        self.h_t, self.w_t = template_img_gray.shape
        self.margin = faceMargin
        self.w_match, self.h_match = faceMargin * 2, faceMargin * 2
        self.last_angle, self.smooth_dx, self.smooth_dy = 0.0, 0.0, 0.0
        self.num_angles, self.local_size = numAngles, 256

        # Compilação
        self.prg = cl.Program(self.ctx, self.KERNEL_SOURCE).build()

        # Formatos: R FLOAT (Cinza) e RGBA UNORM_INT8 (Cores)
        self.fmt_gray = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        self.fmt_rgba = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)

        # Buffers Fixos para evitar realocação
        self.template_gpu = cl.Image(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                                    self.fmt_gray, shape=(self.w_t, self.h_t), 
                                    hostbuf=template_img_gray.astype(np.float32))
        
        self.frame_gray_gpu = cl.Image(self.ctx, self.mf.READ_ONLY, self.fmt_gray, shape=(self.w_f, self.h_f))
        self.frame_rgba_gpu = cl.Image(self.ctx, self.mf.READ_ONLY, self.fmt_rgba, shape=(self.w_f, self.h_f))
        self.output_rgba_gpu = cl.Image(self.ctx, self.mf.WRITE_ONLY, self.fmt_rgba, shape=(self.w_f, self.h_f))
        
        self.cost_map_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, self.w_match * self.h_match * self.num_angles * 4)
        self.res_x_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, 4)
        self.res_y_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, 4)
        self.res_angle_gpu = cl.Buffer(self.ctx, self.mf.READ_WRITE, 4)



    def updtate_template(self, new_template_gray):
        cl.enqueue_copy(self.queue, self.template_gpu, new_template_gray.astype(np.float32), origin=(0,0), region=(self.w_t, self.h_t))

    def reset(self, new_x0, new_y0):
        self.x0, self.y0 = new_x0, new_y0
        self.last_angle, self.smooth_dx, self.smooth_dy = 0.0, 0.0, 0.0    

    def process(self, frame_gray, frame_bgr, zoom=1):
        # 1. Preparar e fazer upload das imagens
        # Converter BGR do OpenCV para RGBA para a GPU
        frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        
        cl.enqueue_copy(self.queue, self.frame_gray_gpu, frame_gray.astype(np.float32), origin=(0,0), region=(self.w_f, self.h_f))
        cl.enqueue_copy(self.queue, self.frame_rgba_gpu, frame_rgba, origin=(0,0), region=(self.w_f, self.h_f))

        # 2. Janela de busca
        curr_x = self.x0 + int(round(self.smooth_dx))
        curr_y = self.y0 + int(round(self.smooth_dy))
        start_x = max(0, min(curr_x - self.margin, self.w_f - self.w_t - self.w_match))
        start_y = max(0, min(curr_y - self.margin, self.h_f - self.h_t - self.h_match))

        # 3. Matching (em Cinza)
        angle_step = np.float32(np.radians(5))
        start_angle = np.float32(self.last_angle - (self.num_angles // 2) * angle_step)
        
        self.prg.match_template_multi_dims(
            self.queue, (self.w_match, self.h_match, self.num_angles), None,
            self.frame_gray_gpu, self.template_gpu, self.cost_map_gpu,
            cl.cltypes.make_int2(self.w_f, self.h_f), cl.cltypes.make_int2(self.w_t, self.h_t),
            np.int32(start_x), np.int32(start_y), start_angle, angle_step, 
            np.float32(1.0), np.float32(0.0), np.int32(1) # Escala fixa p/ estabilização
        )

        # 4. Redução para achar o melhor match
        area_per_angle = np.int32(self.w_match * self.h_match)
        self.prg.find_min_cost_3d(self.queue, (self.local_size,), (self.local_size,),
            self.cost_map_gpu, self.res_x_gpu, self.res_y_gpu, self.res_angle_gpu,
            np.int32(self.w_match), area_per_angle, np.int32(area_per_angle * self.num_angles),
            cl.LocalMemory(self.local_size * 4), cl.LocalMemory(self.local_size * 4))

        # 5. Ler resultados
        x_loc, y_loc, a_idx = np.empty(1, np.int32), np.empty(1, np.int32), np.empty(1, np.int32)
        cl.enqueue_copy(self.queue, x_loc, self.res_x_gpu)
        cl.enqueue_copy(self.queue, y_loc, self.res_y_gpu)
        cl.enqueue_copy(self.queue, a_idx, self.res_angle_gpu)

        self.last_angle = start_angle + (a_idx[0] * angle_step)
        dx, dy = (start_x + x_loc[0] - self.x0), (start_y + y_loc[0] - self.y0)
        
        alpha = 0.2
        self.smooth_dx = alpha * dx + (1 - alpha) * self.smooth_dx
        self.smooth_dy = alpha * dy + (1 - alpha) * self.smooth_dy

        # --- 6. Estabilização em RGB (na GPU) ---
        offset = cl.cltypes.make_int2(np.int32(round(self.smooth_dx)), np.int32(round(self.smooth_dy)))
        self.prg.stabilize(self.queue, (self.w_f, self.h_f), None, 
                        self.frame_rgba_gpu, self.output_rgba_gpu, offset, np.float32(zoom))

        # --- 7. Download do Resultado (CORRIGIDO) ---
        res_rgba = np.empty((self.h_f, self.w_f, 4), dtype=np.uint8)

        # Para cl.Image, origin e region são OBRIGATÓRIOS no enqueue_copy
        cl.enqueue_copy(
            self.queue, 
            res_rgba, 
            self.output_rgba_gpu, 
            origin=(0, 0), 
            region=(self.w_f, self.h_f)
        )

        return cv2.cvtColor(res_rgba, cv2.COLOR_RGBA2BGR), start_x + x_loc[0], start_y + y_loc[0], np.degrees(self.last_angle)
    

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