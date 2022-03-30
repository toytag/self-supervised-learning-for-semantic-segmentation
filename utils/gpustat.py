import time
from gpustat.core import GPUStatCollection

# Check GPU status
def get_gpu_status():
    gpus_stats = GPUStatCollection.new_query()

    info = gpus_stats.jsonify()["gpus"]
    gpu_list = []

    mem_ratio_threshold = 0.05
    util_ratio_threshold = 5
    for idx, each in enumerate(info):
        mem_ratio = each["memory.used"] / each["memory.total"]
        util_ratio = each["utilization.gpu"]
        if mem_ratio < mem_ratio_threshold and util_ratio < util_ratio_threshold:
            gpu_list.append(idx)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]: Scan GPUs to get {len(gpu_list)} free GPU")

    return gpu_list
