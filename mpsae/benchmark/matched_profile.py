"""Stage-local CUDA peaks and process RSS sampled by a background thread."""
import gc
import threading
import time


class Stage:
    def __init__(self,device):
        import torch
        self.device=torch.device(device)
        self.result={}

    def __enter__(self):
        import psutil
        import torch
        gc.collect()
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.result['gpu_baseline_allocated_bytes']=torch.cuda.memory_allocated(self.device)
        self.process=psutil.Process()
        self.rss=self.process.memory_info().rss
        self.result['rss_baseline_bytes']=self.rss
        self.stop=threading.Event()
        def poll():
            while not self.stop.wait(.1):
                self.rss=max(self.rss,self.process.memory_info().rss)
        self.thread=threading.Thread(target=poll,daemon=True)
        self.thread.start()
        self.started=time.perf_counter()
        return self.result

    def __exit__(self,*exc):
        import torch
        try:
            if self.device.type=='cuda':
                torch.cuda.synchronize(self.device)
            self.result['wall_seconds']=time.perf_counter()-self.started
            self.result['gpu_peak_allocated_bytes']=(torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None)
            self.result['gpu_peak_reserved_bytes']=(torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None)
            self.result['rss_peak_sampled_bytes']=max(self.rss,self.process.memory_info().rss)
            self.result['rss_sampling_seconds']=.1
        finally:
            self.stop.set()
            self.thread.join()


def measure(search,device,config,queries):
    """Five warm-ups and five measurements by default; CPU smoke never claims GPU speed."""
    import torch
    from ..evaluate.timing import _measure, _per_query
    if torch.device(device).type!='cuda':
        return {'status':'unavailable','reason':'CPU smoke test; no comparable GPU timing'}
    result=_per_query(_measure(search,device=torch.device(device),warmup=config.warmup,
                               rounds=config.rounds),queries)
    result['status']='complete'
    return result
