"""Dependency-free browser viewer for the local pose WebSocket."""
from __future__ import annotations

import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

_HTML = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MobilePose viewer</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101318;color:#e9edf2;font:14px system-ui,sans-serif;overflow:hidden}header{height:52px;padding:12px 18px;border-bottom:1px solid #2b313b;display:flex;gap:24px;align-items:center}h1{font-size:16px;margin:0}.ok{color:#8bd5a4}.bad{color:#f28b82}main{height:calc(100vh - 52px);display:grid;grid-template-columns:minmax(0,1fr) 280px}canvas{width:100%;height:100%;display:block;background:#171b22}aside{border-left:1px solid #2b313b;padding:16px;line-height:1.8;overflow:auto}dt{color:#8d98a8;font-size:12px}dd{margin:0 0 8px}button{background:#252c37;color:#e9edf2;border:1px solid #3a4553;border-radius:4px;padding:6px 9px;cursor:pointer;margin:2px}input{width:100%;background:#171b22;color:#e9edf2;border:1px solid #3a4553;padding:6px;border-radius:3px}pre{white-space:pre-wrap;font-size:11px;line-height:1.35;color:#b8c3d1}@media(max-width:700px){main{grid-template-columns:1fr}aside{position:absolute;right:0;top:52px;width:220px;background:#101318ee}}</style>
<header><h1>MobilePose realtime viewer</h1><span id="status" class="bad">connecting...</span><span id="recordState"></span></header><main><canvas id="view"></canvas><aside><dl><dt>Frame</dt><dd id="frame">-</dd><dt>Layout</dt><dd id="layout">-</dd><dt>FPS</dt><dd id="fps">-</dd><dt>Model lag</dt><dd id="lag">-</dd><dt>Foot contact</dt><dd id="contact">-</dd><dt>Recorded frames</dt><dd id="recorded">0</dd></dl><button id="front">Front</button> <button id="side">Side</button><hr><button id="record">Start recording</button> <button id="stop">Stop</button><button id="replay">Replay</button><button id="clear">Clear</button><button id="download">Download JSONL</button><hr><strong>Feature comparison</strong><input id="capturePath" placeholder="D:\\...\\data\\raw\\ours\\3_站位\\1"><button id="compare">Compare offline / realtime</button><pre id="compareResult">Enter a capture directory and compare.</pre></aside></main>
<script>
const c=document.getElementById('view'),x=c.getContext('2d'),statusEl=document.getElementById('status'),fpsEl=document.getElementById('fps');let mode='front',last=0,n=0,frame=null;const par=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],off=[[0,0,0],[-.18,-.08,0],[.18,-.08,0],[0,.22,0],[0,-.42,0],[0,-.42,0],[0,.24,0],[0,-.42,0],[0,-.42,0],[0,.22,0],[0,-.13,.08],[0,-.13,.08],[0,.20,0],[-.12,.12,0],[.12,.12,0],[0,.25,0],[-.2,0,0],[.2,0,0],[-.3,0,0],[.3,0,0],[-.24,0,0],[.24,0,0],[-.12,0,.03],[.12,0,.03]];
function qm(a,b){return[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]]}function qr(q,v){let r=qm(qm(q,[...v,0]),[-q[0],-q[1],-q[2],q[3]]);return r.slice(0,3)}
function draw(m){frame=m;let q=Array.from({length:24},(_,i)=>m.joints.slice(4*i,4*i+4)),g=[],p=[];for(let i=0;i<24;i++){let a=par[i];g[i]=a<0?q[i]:qm(g[a],q[i]);p[i]=a<0?[0,0,0]:p[a].map((v,j)=>v+qr(g[a],off[i])[j])}let w=c.clientWidth,h=c.clientHeight,d=devicePixelRatio||1;if(c.width!==w*d||c.height!==h*d){c.width=w*d;c.height=h*d}x.setTransform(d,0,0,d,0,0);x.clearRect(0,0,w,h);let lo=Math.min(...p.map(v=>v[1])),hi=Math.max(...p.map(v=>v[1])),s=Math.min(w*.38/(hi-lo+.1),h*.72/(hi-lo+.1));let xy=v=>[w/2+(mode==='front'?v[0]:v[2])*s,h*.74-v[1]*s];x.strokeStyle='#70b7ff';x.lineWidth=5;x.lineCap='round';for(let i=1;i<24;i++){let a=xy(p[par[i]]),b=xy(p[i]);x.beginPath();x.moveTo(...a);x.lineTo(...b);x.stroke()}x.fillStyle='#f2c14e';p.forEach((v,i)=>{let a=xy(v);x.beginPath();x.arc(...a,i?5:7,0,7);x.fill()});document.getElementById('frame').textContent=m.frame;document.getElementById('layout').textContent=m.layout||'-';document.getElementById('lag').textContent=(m.modelLagFrames??'-')+' frames';document.getElementById('contact').textContent=m.footContact?m.footContact.map(v=>Number(v).toFixed(2)).join(' / '):(m.footContactEnabled?'waiting':'disabled')}
let recording=false,recorded=[],replayTimer=null;const recordState=document.getElementById('recordState'),recordedEl=document.getElementById('recorded'),compareResult=document.getElementById('compareResult');function setRecording(v){recording=v;recordState.textContent=v?' recording':' ';recordState.className=v?'ok':'';document.getElementById('record').disabled=v}function stopReplay(){if(replayTimer){clearInterval(replayTimer);replayTimer=null}}function connect(){let ws=new WebSocket('ws://'+location.hostname+':__WS_PORT__');ws.onopen=()=>{statusEl.textContent='connected';statusEl.className='ok'};ws.onclose=()=>{statusEl.textContent='disconnected';statusEl.className='bad';setTimeout(connect,1000)};ws.onerror=()=>ws.close();ws.onmessage=e=>{let t=performance.now();n++;if(t-last>1000){fpsEl.textContent=(n*1000/(t-last)).toFixed(1);n=0;last=t}let m=JSON.parse(e.data);if(recording)recorded.push(m);recordedEl.textContent=recorded.length;draw(m)}};document.getElementById('front').onclick=()=>{mode='front';if(frame)draw(frame)};document.getElementById('side').onclick=()=>{mode='side';if(frame)draw(frame)};document.getElementById('record').onclick=()=>{stopReplay();recorded=[];recordedEl.textContent='0';setRecording(true)};document.getElementById('stop').onclick=()=>setRecording(false);document.getElementById('clear').onclick=()=>{stopReplay();setRecording(false);recorded=[];recordedEl.textContent='0'};document.getElementById('replay').onclick=()=>{if(!recorded.length)return;stopReplay();setRecording(false);let i=0;replayTimer=setInterval(()=>{if(i>=recorded.length){stopReplay();return}draw(recorded[i++])},1000/30)};document.getElementById('download').onclick=()=>{if(!recorded.length)return;let body=recorded.map(v=>JSON.stringify(v)).join('\n')+'\n',a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/jsonl'}));a.download='mobilepose-pose-'+new Date().toISOString().replace(/[:.]/g,'-')+'.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};document.getElementById('compare').onclick=async()=>{let path=document.getElementById('capturePath').value.trim();if(!path){compareResult.textContent='Enter a capture directory.';return}compareResult.textContent='Comparing...';try{let res=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({capture:path})});let data=await res.json();compareResult.textContent=data.error||JSON.stringify(data,null,2)}catch(err){compareResult.textContent=String(err)}};onresize=()=>frame&&draw(frame);connect();
</script>'''

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        html = _HTML.replace("__WS_PORT__", str(self.server.ws_port))
        html = html.replace(
            "</aside>",
            """<hr><strong>Runtime constraints</strong>
<div id=\"constraints\"><label><input type=\"checkbox\" data-constraint=\"useTrackerAnchors\"> Tracker anchors</label>
<label><input type=\"checkbox\" data-constraint=\"enforceJointLimits\"> Joint limits</label>
<label><input type=\"checkbox\" data-constraint=\"guardKneeHyperextension\"> Knee guard</label>
<label><input type=\"checkbox\" data-constraint=\"useTemporalSmoothing\"> Temporal smoothing</label>
<label><input type=\"checkbox\" data-constraint=\"enforceBoneLengths\"> Bone lengths</label>
<label><input type=\"checkbox\" data-constraint=\"useKneeDirectionConstraint\"> Knee direction</label>
<label><input type=\"checkbox\" data-constraint=\"useFootIk\"> Contact foot IK</label>
<label><input type=\"checkbox\" data-constraint=\"useRelativeFootGrounding\"> Foot grounding</label>
<label>Anchor <input type=\"range\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.65\" data-constraint=\"trackerAnchorWeight\"></label>
<label>Smoothing <input type=\"range\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.35\" data-constraint=\"temporalSmoothing\"></label>
<label>IK weight <input type=\"range\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.8\" data-constraint=\"footIkWeight\"></label>
<button id=\"resetConstraints\">Reset constraints</button></div></aside>""",
        )
        html = html.replace(
            "let recording=false,recorded=[],replayTimer=null;",
            "let ws=null;let recording=false,recorded=[],replayTimer=null;",
        )
        html = html.replace("function connect(){let ws=new WebSocket", "function connect(){ws=new WebSocket")
        html = html.replace(
            "let m=JSON.parse(e.data);if(recording)",
            "let m=JSON.parse(e.data);if(m.type==='constraints'){document.querySelectorAll('[data-constraint]').forEach(el=>{if(m[el.dataset.constraint]!==undefined)el[el.type==='checkbox'?'checked':'value']=m[el.dataset.constraint]});return}if(recording)",
        )
        html = html.replace(
            "</script>",
            """document.querySelectorAll('[data-constraint]').forEach(el=>el.onchange=()=>{if(ws&&ws.readyState===1){let c={type:'constraints'};document.querySelectorAll('[data-constraint]').forEach(v=>c[v.dataset.constraint]=v.type==='checkbox'?v.checked:Number(v.value));ws.send(JSON.stringify(c))}});document.getElementById('resetConstraints').onclick=()=>{document.querySelectorAll('[data-constraint]').forEach(el=>el[el.type==='checkbox'?'checked':'value']=el.type==='checkbox'?false:(el.dataset.constraint==='trackerAnchorWeight'?0.65:el.dataset.constraint==='temporalSmoothing'?0.35:0.8));document.querySelector('[data-constraint=useTemporalSmoothing]').dispatchEvent(new Event('change'))};</script>""",
        )
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/compare":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size).decode("utf-8"))
            from mobileposer.realtime.feature_compare import compare_capture
            result = compare_capture(request["capture"])
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            status = 200
        except Exception as exc:  # return diagnostics to the local page
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
            status = 400
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass

def start_web_visualizer(host: str, port: int, websocket_port: int):
    server = ThreadingHTTPServer((host, port), _Handler)
    server.ws_port = websocket_port
    threading.Thread(target=server.serve_forever, name="mobilepose-web", daemon=True).start()
    return server
