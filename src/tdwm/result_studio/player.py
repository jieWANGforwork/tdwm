"""Self-contained synchronized frame/coordinate viewer. Never draws fake RGB."""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .comparison import COMPARISON_CSS, BaselineComparison, comparison_html
from .models import Trajectory

COLORS = ["#6A7B93", "#2457DB", "#BA6B25", "#8654C7"]
PREVIEW_SIZE = 224
CARD_WIDTH = 256


def image_uri(path: str):
    with Image.open(path) as image:
        if image.format == "JPEG" and image.width <= 360 and image.height <= 300:
            # Preserve the real Cube dataset's original JPEG payload in the UI.
            return (
                "data:image/jpeg;base64,"
                + base64.b64encode(Path(path).read_bytes()).decode()
            )
        image.thumbnail((360, 300))
        output = io.BytesIO()
        image.convert("RGB").save(output, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def player_html(
    tracks: list[Trajectory],
    demo=False,
    comparisons: list[BaselineComparison | None] | None = None,
    baseline_label: str = "",
):
    if comparisons is not None and len(comparisons) != len(tracks):
        raise ValueError("Baseline 对比标签必须与轨迹逐一对应")
    payload = []
    for i, track in enumerate(tracks):
        # Only the selected trial is decoded. Hard cap is explicit in the UI.
        frame_indexes = (
            np.linspace(
                0, len(track.frames) - 1, min(201, len(track.frames)), dtype=int
            )
            if track.frames
            else []
        )
        frames = [image_uri(track.frames[int(index)]) for index in frame_indexes]
        coords = (
            track.states[:, :2].tolist()
            if track.states is not None and track.states.shape[1] >= 2
            else []
        )
        payload.append(
            {
                "name": track.name,
                "kind": track.kind,
                "status": track.success,
                "color": COLORS[i % len(COLORS)],
                "coords": coords,
                "frames": frames,
                "indexes": list(map(int, frame_indexes)),
                "length": track.length,
                "comparison_html": comparison_html(comparisons[i], baseline_label)
                if comparisons is not None
                and comparisons[i] is not None
                and track.kind != "reference"
                else "",
            }
        )
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    return (
        TEMPLATE.replace("__TRACKS__", encoded)
        .replace("__DEMO__", "true" if demo else "false")
        .replace("__PREVIEW_SIZE__", str(PREVIEW_SIZE))
        .replace("__CARD_WIDTH__", str(CARD_WIDTH))
        .replace("__BASELINE_CSS__", COMPARISON_CSS)
    )


TEMPLATE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><style>
*{box-sizing:border-box}
body{margin:0;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#16263e;background:transparent}
__BASELINE_CSS__
button,select,input{font:inherit}button{cursor:pointer}
.grid{display:flex;flex-wrap:wrap;justify-content:flex-start;align-items:flex-start;gap:16px}
.card{flex:0 0 __CARD_WIDTH__px;width:__CARD_WIDTH__px;max-width:100%;background:white;border:1px solid #dee5f0;border-radius:12px;overflow:hidden}
.title{padding:15px 16px 12px;font-weight:650;display:flex;justify-content:space-between;gap:8px;min-height:47px}
.title>span:first-child{min-width:0;overflow-wrap:anywhere}.kind{font-size:12px;color:#77849a;font-weight:400;flex-shrink:0}
canvas{display:block;width:__PREVIEW_SIZE__px;max-width:100%;height:__PREVIEW_SIZE__px;margin:0 auto;background:#f0f4fa}
.foot{padding:13px 16px;display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap}
.badge{padding:4px 9px;border-radius:5px;font-size:13px;background:#edf1f7;color:#52627a}.yes{color:#167252;background:#e6f4ed}.no{color:#b3444d;background:#fff0f1}.hint{color:#7b8799;font-size:12px}
.toolbar{display:flex;align-items:center;gap:14px;margin:18px 0 8px;padding:12px 14px;border:1px solid #dee5f0;background:#fff;border-radius:10px}
.toolbar button{border:0;background:#2457db;color:#fff;padding:9px 18px;border-radius:7px;min-width:90px}
input[type=range]{flex:1;min-width:45px;accent-color:#2457db}.counter{font-variant-numeric:tabular-nums;white-space:nowrap;color:#41536b}select{border:1px solid #dce2eb;border-radius:5px;padding:6px;background:white}.legend{font-size:12px;color:#738298;line-height:1.8}
@media(max-width:680px){.grid{gap:8px}.toolbar{gap:6px;padding:8px}.toolbar button{min-width:65px;padding:9px}}
</style></head><body><div id="grid" class="grid"></div><div class="toolbar"><button id="play">播放全部</button><input id="time" aria-label="同步时间轴" type="range" min="0" max="1000" value="0"><span id="counter" class="counter">0%</span><select id="speed" aria-label="播放速度"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><button id="reset" style="background:#edf1f8;color:#40516a;min-width:55px">重置</button></div><div class="legend" id="legend"></div><script>
const tracks=__TRACKS__,demo=__DEMO__,images=[];let progress=0,playing=false,last=0;
const grid=document.getElementById('grid');
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
tracks.forEach((t,i)=>{const label=t.kind==='reference'?'参考路径':t.kind==='predicted'?'预测轨迹':t.status===true?'成功':t.status===false?'失败':'未标注';const state=t.kind==='executed'?(t.status===true?'yes':t.status===false?'no':''):'';grid.insertAdjacentHTML('beforeend',`<div class="card"><div class="title"><span>${escape(t.name)}</span><span class="kind">${t.kind==='reference'?'TARGET':t.kind==='predicted'?'PREDICTED':'EXECUTED'}</span></div><canvas id="c${i}"></canvas><div class="foot"><span class="badge ${state}">${label}${demo&&t.kind==='executed'?' · 演示':''}</span><span class="hint" id="t${i}"></span></div>${t.comparison_html}</div>`);images[i]=t.frames.map(src=>{const im=new Image();im.src=src;im.onload=draw;return im})});
const all=tracks.flatMap(t=>t.coords).filter(p=>p.every(Number.isFinite));let bounds=[0,1,0,1];if(all.length){bounds=[Math.min(...all.map(p=>p[0])),Math.max(...all.map(p=>p[0])),Math.min(...all.map(p=>p[1])),Math.max(...all.map(p=>p[1]))]};
function draw(){tracks.forEach((t,i)=>{const c=document.getElementById('c'+i),r=c.getBoundingClientRect(),w=r.width,h=r.height,dpr=devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const idx=Math.round(progress*Math.max(0,t.length-1));document.getElementById('t'+i).textContent=t.length?`帧 ${idx} / ${t.length-1}`:'未接入轨迹';
if(images[i].length){let at=0;for(let j=1;j<t.indexes.length;j++){if(Math.abs(t.indexes[j]-idx)<Math.abs(t.indexes[at]-idx))at=j};const im=images[i][at];if(im.complete&&im.naturalWidth){const scale=Math.min(1,w/im.width,h/im.height);ctx.drawImage(im,(w-im.width*scale)/2,(h-im.height*scale)/2,im.width*scale,im.height*scale)}return}
if(!t.coords.length){ctx.fillStyle='#7b899b';ctx.font='14px sans-serif';ctx.textAlign='center';ctx.fillText('完整轨迹待接入',w/2,h/2-8);ctx.font='12px sans-serif';ctx.fillText('不会用起终点插值冒充真实路径',w/2,h/2+17);return}
const map=p=>[28+(p[0]-bounds[0])/(bounds[1]-bounds[0]||1)*(w-56),h-26-(p[1]-bounds[2])/(bounds[3]-bounds[2]||1)*(h-55)];ctx.strokeStyle='#e1e7f0';ctx.lineWidth=1;for(let j=1;j<6;j++){ctx.beginPath();ctx.moveTo(w*j/6,15);ctx.lineTo(w*j/6,h-15);ctx.stroke();ctx.beginPath();ctx.moveTo(15,h*j/6);ctx.lineTo(w-15,h*j/6);ctx.stroke()}
function path(points,color,width){ctx.beginPath();points.forEach((p,j)=>{const q=map(p);j?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineJoin='round';ctx.stroke()};path(t.coords,t.color+'35',3);path(t.coords.slice(0,idx+1),t.color,3);const target=map(tracks[0].coords.at(-1)||t.coords.at(-1));ctx.strokeStyle='#50637c';ctx.lineWidth=2;ctx.strokeRect(target[0]-5,target[1]-5,10,10);const start=map(t.coords[0]);ctx.fillStyle='white';ctx.beginPath();ctx.arc(...start,5,0,7);ctx.fill();ctx.stroke();const here=map(t.coords[Math.min(idx,t.coords.length-1)]);ctx.shadowColor=t.color+'77';ctx.shadowBlur=10;ctx.fillStyle=t.color;ctx.beginPath();ctx.arc(...here,6,0,7);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#7c889a';ctx.font='11px sans-serif';ctx.fillText(demo?'合成状态坐标 · 非机器人图像':'状态 d0 / d1 投影',14,18)
});document.getElementById('counter').textContent=Math.round(progress*100)+'%';document.getElementById('time').value=Math.round(progress*1000)}
document.getElementById('legend').textContent='各列按相对进度同步，不代表相同物理时刻。'+(demo?' 淡线：完整路径 · 实线：已播放路径 · 方框：参考终点。': ' 长于 201 帧的图像序列在同步预览中抽样；完整文件保留。');
function tick(ts){if(playing){progress=Math.min(1,progress+(ts-last)/8000*Number(document.getElementById('speed').value));draw();if(progress>=1){playing=false;document.getElementById('play').textContent='重新播放'}}last=ts;requestAnimationFrame(tick)}
document.getElementById('play').onclick=()=>{if(progress>=1)progress=0;playing=!playing;document.getElementById('play').textContent=playing?'暂停全部':'播放全部';draw()};document.getElementById('time').oninput=e=>{progress=e.target.value/1000;draw()};document.getElementById('reset').onclick=()=>{progress=0;playing=false;document.getElementById('play').textContent='播放全部';draw()};window.addEventListener('resize',draw);draw();requestAnimationFrame(tick);
if(document.modelContext?.registerTool){const lifecycle=new AbortController();document.modelContext.registerTool({name:'set_trajectory_progress',description:'Set the shared playback progress of the visible trajectory comparison.',inputSchema:{type:'object',properties:{progress:{type:'number',minimum:0,maximum:1}},required:['progress'],additionalProperties:false},execute(input){if(typeof input.progress!=='number'||!Number.isFinite(input.progress)||input.progress<0||input.progress>1)throw Error('progress must be between 0 and 1');progress=input.progress;draw();return {progress,tracks:tracks.map(t=>t.name)};}},{signal:lifecycle.signal});window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true})}
</script></body></html>"""
