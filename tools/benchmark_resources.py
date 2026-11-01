"""Measure real detector/OCR timings and sampled process RAM/device VRAM."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from PIL import Image

from vespa_retrieval.ocr_backends import infer
from vespa_retrieval.pipeline import PROJECT_ROOT, detect
from vespa_retrieval.pipeline_utils import write_json
from vespa_retrieval.server import local_server
from vespa_retrieval.run_qwen_layout_ocr import TEXT_PROMPT, TABLE_PROMPT


def gpu():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    name, total, used, driver = [part.strip() for part in line.split(',')]
    return {'name':name,'total_mib':int(total),'used_mib':int(used),'driver':driver}


class Measurement:
    def __enter__(self):
        self.samples=[]; self.stop=threading.Event(); self.started=time.perf_counter()
        self.thread=threading.Thread(target=self.sample,daemon=True); self.thread.start()
        return self

    def sample(self):
        while not self.stop.is_set():
            process=psutil.Process()
            rss=0
            for proc in [process]+process.children(recursive=True):
                try:
                    rss+=proc.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    pass
            self.samples.append({'time_seconds':time.perf_counter()-self.started,'rss_mib':rss/1048576,'gpu_used_mib':gpu()['used_mib']})
            self.stop.wait(0.25)

    def __exit__(self,*args):
        self.elapsed=time.perf_counter()-self.started
        self.stop.set(); self.thread.join()

    def result(self):
        return {'wall_seconds':self.elapsed,'sample_count':len(self.samples),
                'peak_process_tree_rss_mib':max(s['rss_mib'] for s in self.samples),
                'peak_device_used_mib':max(s['gpu_used_mib'] for s in self.samples),'samples':self.samples}


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    report={'date':time.strftime('%Y-%m-%d'),'python':sys.version,'platform':platform.platform(),
            'cpu':next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
            'logical_cpus':psutil.cpu_count(),'host_ram_mib':psutil.virtual_memory().total/1048576,
            'gpu_baseline':gpu(),'method':'250ms wait between samples plus nvidia-smi overhead. RSS sums benchmark Python and descendants; shared mappings may be counted more than once. VRAM is whole-device used memory, not allocator high-water mark.',
            'configuration':{'detector_device':'cpu','imgsz':1024,'confidence':0.25,'dedup_iou':0.20,'ocr':'Qwen3-VL-4B-Instruct Q8_0 + f16 projector','context':8192,'slots':1,'gpu_layers':99,'flash_attention':True},'ocr_requests':[]}
    with Measurement() as measurement:
        detect(PROJECT_ROOT/'imgs',output,'juliozhao/DocLayout-YOLO-DocStructBench')
    report['detection']=measurement.result()
    layouts=json.loads((output/'layouts.json').read_text())
    report['detection']['pages']=len(layouts)
    report['detection']['boxes']=sum(len(boxes) for boxes in layouts.values())
    report['detection']['page_images']=[{'name':p.name,'size':list(Image.open(p).size)} for p in sorted((PROJECT_ROOT/'imgs').glob('*.png'))]
    write_json(output/'results.json',report)
    # Existing unchanged detector crops isolate OCR from layout/crop creation.
    cases=[('cover_logo','vespa_manual_0/vespa_manual_0_2.png',TEXT_PROMPT,4096),
           ('body_text','vespa_manual_1/vespa_manual_1_2.png',TEXT_PROMPT,4096),
           ('lubrication_table','vespa_manual_29/vespa_manual_29_1.png',TABLE_PROMPT,6144),
           ('troubleshooting_table','vespa_manual_32/vespa_manual_32_0.png',TABLE_PROMPT,6144)]
    manager=local_server(PROJECT_ROOT/'vendor/llama.cpp/build/bin/llama-server',
                         PROJECT_ROOT/'models/logo_ocr/Qwen3-VL-4B-Instruct.Q8_0.gguf',
                         PROJECT_ROOT/'models/logo_ocr/Qwen3-VL-4B-Instruct.mmproj-f16.gguf',output/'ocr_server.log')
    try:
        with Measurement() as startup:
            endpoint=manager.__enter__()
        report['startup']=startup.result()
        for name,relative,prompt,budget in cases:
            image=PROJECT_ROOT/'outputs/qwen_pipeline/crops'/relative
            print('Measuring OCR',name,flush=True)
            with Measurement() as measurement:
                result=infer(image,'qwen3vl4b',endpoint=endpoint,prompt=prompt,max_tokens=budget,timeout=600)
            record={'case':name,'crop':relative,'image_size':list(Image.open(image).size),
                    'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                    'measurement':measurement.result(),'result':result}
            report['ocr_requests'].append(record)
            write_json(output/'results.json',report)
    finally:
        manager.__exit__(None,None,None)
    report['gpu_after_shutdown']=gpu()
    write_json(output/'results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('gpu_baseline','cpu','host_ram_mib')},indent=2))
    for name in ('detection','startup'):
        print(name,{k:v for k,v in report[name].items() if k not in ('samples','page_images')})
    for record in report['ocr_requests']:
        print(record['case'],{k:v for k,v in record['measurement'].items() if k!='samples'},record['result']['response'].get('usage'),record['result']['response']['choices'][0]['finish_reason'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())