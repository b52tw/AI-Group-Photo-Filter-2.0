# -*- coding: utf-8 -*-
import os, sys, csv, shutil, queue, threading, hashlib, sqlite3
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2, numpy as np
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()

APP='峻爸 AI智慧照片分類器 v2.0'
OWNER='製作人／程式所有人：峻爸'
VER='2.0.0'
EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.heic','.heif','.tif','.tiff'}
YUNET='face_detection_yunet_2023mar.onnx'
ARCFACE='w600k_r50.onnx'
YOLO='yolov8n.onnx'
COCO=['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
ZH={'person':'人物','bicycle':'自行車','car':'汽車','motorcycle':'機車','airplane':'飛機','bus':'公車','train':'火車','truck':'卡車','boat':'船','traffic light':'紅綠燈','bird':'鳥','cat':'貓','dog':'狗','horse':'馬','backpack':'背包','umbrella':'雨傘','handbag':'手提包','tie':'領帶','suitcase':'行李箱','sports ball':'球類','baseball bat':'棒球棒','baseball glove':'棒球手套','tennis racket':'網球拍','bottle':'瓶子','wine glass':'酒杯','cup':'杯子','fork':'叉子','knife':'刀具','spoon':'湯匙','bowl':'碗','banana':'香蕉','apple':'蘋果','sandwich':'三明治','orange':'橘子','pizza':'披薩','cake':'蛋糕','chair':'椅子','couch':'沙發','potted plant':'盆栽','bed':'床','dining table':'餐桌','toilet':'馬桶','tv':'螢幕／電視','laptop':'筆電','keyboard':'鍵盤','cell phone':'手機','microwave':'微波爐','oven':'烤箱','sink':'水槽','refrigerator':'冰箱','book':'書籍','clock':'時鐘'}
INDOOR={'chair','couch','bed','dining table','toilet','tv','laptop','keyboard','cell phone','microwave','oven','sink','refrigerator','book','clock'}
FOOD={'dining table','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','pizza','cake'}
SPORT={'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket'}
TRAFFIC={'bicycle','car','motorcycle','bus','train','truck','traffic light','stop sign','parking meter'}
NATURE={'bird','horse','sheep','cow','elephant','bear','zebra','giraffe'}

def res(name): return Path(getattr(sys,'_MEIPASS',Path(__file__).parent))/name

def appdata():
    p=Path(os.getenv('LOCALAPPDATA') or Path.home())/'JunBaPhotoAI'; p.mkdir(parents=True,exist_ok=True); return p

def load_img(p,maxdim=1600):
    with Image.open(p) as im:
        ex=im.getexif(); dt=''
        for k in (36867,36868,306):
            v=ex.get(k)
            if v:
                try: dt=datetime.strptime(str(v),'%Y:%m:%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'); break
                except: pass
        im=ImageOps.exif_transpose(im).convert('RGB'); w,h=im.size
        if max(w,h)>maxdim:
            r=maxdim/max(w,h); im=im.resize((int(w*r),int(h*r)),Image.Resampling.LANCZOS)
        arr=np.asarray(im)
    if not dt:
        try: dt=datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        except: pass
    return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR),dt

def norm(x):
    x=np.asarray(x,dtype=np.float32).reshape(-1); n=np.linalg.norm(x); return x/(n+1e-9)

def refsig(paths):
    h=hashlib.sha1()
    for p in sorted(paths):
        try:
            s=Path(p).stat(); h.update(f'{p}|{s.st_size}|{s.st_mtime_ns}'.encode())
        except: h.update(str(p).encode())
    return h.hexdigest()

def unique(folder,p):
    folder.mkdir(parents=True,exist_ok=True); d=folder/p.name; n=2
    while d.exists(): d=folder/f'{p.stem}_{n}{p.suffix}'; n+=1
    return d

def bucket(n):
    if n<=0:return '00_未偵測到人臉'
    if n==1:return '01_單人照'
    if n==2:return '02_雙人照'
    if n<=5:return '03_小組照_3-5人'
    if n<=10:return '04_團體照_6-10人'
    return '05_大合照_11人以上'

def scene(tags):
    s=set(tags)
    if s&SPORT:return '運動場景'
    if s&FOOD:return '餐飲場景'
    if s&TRAFFIC:return '街景／交通'
    if s&NATURE:return '戶外／自然'
    if s&INDOOR:return '室內'
    if 'person' in s:return '人物活動'
    return '其他／待確認'

class Cache:
    def __init__(self):
        self.c=sqlite3.connect(appdata()/'photo_index_v2.sqlite3',check_same_thread=False); self.lock=threading.Lock()
        self.c.execute('''CREATE TABLE IF NOT EXISTS r(path TEXT,size INTEGER,mtime INTEGER,ver TEXT,mode TEXT,ref TEXT,data TEXT,PRIMARY KEY(path,size,mtime,ver,mode,ref))'''); self.c.commit()
    def get(self,p,mode,ref):
        try:
            st=p.stat(); key=(str(p.resolve()),st.st_size,st.st_mtime_ns,VER,mode,ref)
            with self.lock: row=self.c.execute('SELECT data FROM r WHERE path=? AND size=? AND mtime=? AND ver=? AND mode=? AND ref=?',key).fetchone()
            if row:
                import json; x=json.loads(row[0]); x['cached']=True; return x
        except: pass
    def put(self,p,mode,ref,x):
        try:
            import json; st=p.stat(); key=(str(p.resolve()),st.st_size,st.st_mtime_ns,VER,mode,ref,json.dumps(x,ensure_ascii=False))
            with self.lock: self.c.execute('INSERT OR REPLACE INTO r VALUES(?,?,?,?,?,?,?)',key); self.c.commit()
        except: pass

class Detector:
    def __init__(self,score): self.d=cv2.FaceDetectorYN.create(str(res(YUNET)),'',(320,320),score,.3,5000)
    def faces(self,img):
        h,w=img.shape[:2]; self.d.setInputSize((w,h)); _,f=self.d.detect(img); return [] if f is None else list(f)

class ArcMatcher:
    T=np.array([[73.5318,51.5014],[38.2946,51.6963],[56.0252,71.7366],[70.7299,92.2041],[41.5493,92.3655]],np.float32)
    def __init__(self): self.net=cv2.dnn.readNetFromONNX(str(res(ARCFACE))); self.refs=[]; self.center=None
    def emb(self,img,f):
        pts=np.asarray(f[4:14],np.float32).reshape(5,2); M,_=cv2.estimateAffinePartial2D(pts,self.T,method=cv2.LMEDS)
        if M is None:return None
        crop=cv2.warpAffine(img,M,(112,112)); blob=cv2.dnn.blobFromImage(crop,1/127.5,(112,112),(127.5,127.5,127.5),swapRB=True)
        self.net.setInput(blob); return norm(self.net.forward())
    def quality(self,img,f):
        x,y,w,h=map(float,f[:4]); x=max(0,int(x)); y=max(0,int(y)); x2=min(img.shape[1],int(x+w)); y2=min(img.shape[0],int(y+h)); c=img[y:y2,x:x2]
        if c.size==0:return 0
        blur=cv2.Laplacian(cv2.cvtColor(c,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var(); return .65*min(1,min(w,h)/95)+.35*min(1,blur/120)
    def build(self,paths,det):
        good=[]
        for p in paths:
            try:
                im,_=load_img(Path(p),2200); fs=det.faces(im)
                if fs:
                    e=self.emb(im,max(fs,key=lambda x:x[2]*x[3]));
                    if e is not None:good.append(e)
            except: pass
        self.refs=good; self.center=norm(np.mean(np.stack(good),axis=0)) if good else None; return len(good)
    def score(self,img,faces):
        if self.center is None:return None,'無參考模板'
        best=-9; qbest=0
        for f in faces:
            q=self.quality(img,f)
            if q<.18:continue
            e=self.emb(img,f)
            if e is None:continue
            s=.7*float(np.dot(e,self.center))+.3*max(float(np.dot(e,r)) for r in self.refs)
            if s>best:best=s;qbest=q
        if best<-1:return None,'人臉太小／模糊，待人工確認'
        return best,f'臉部品質 {qbest:.2f}'

class Tags:
    def __init__(self): self.net=cv2.dnn.readNetFromONNX(str(res(YOLO)))
    def run(self,img):
        self.net.setInput(cv2.dnn.blobFromImage(img,1/255,(640,640),swapRB=True)); a=np.squeeze(self.net.forward())
        if a.ndim!=2:return []
        if a.shape[0]<=100:a=a.T
        found={}
        for row in a:
            sc=row[4:]; cid=int(np.argmax(sc)); v=float(sc[cid])
            if v>=.32 and cid<len(COCO): found[COCO[cid]]=max(found.get(COCO[cid],0),v)
        return [k for k,v in sorted(found.items(),key=lambda z:z[1],reverse=True)][:12]

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title(f'{APP}｜{OWNER}'); self.geometry('1180x790'); self.minsize(980,680)
        self.src=tk.StringVar(); self.dst=tk.StringVar(); self.mode=tk.StringVar(value='標準模式'); self.sens=tk.StringVar(value='平衡'); self.strict=tk.StringVar(value='標準')
        self.people=tk.BooleanVar(value=True); self.date=tk.BooleanVar(value=True); self.tags=tk.BooleanVar(value=True); self.usecache=tk.BooleanVar(value=True)
        self.pf=tk.StringVar(value='全部'); self.sf=tk.StringVar(value='全部'); self.df=tk.StringVar(); self.kw=tk.StringVar(); self.onlymatch=tk.BooleanVar(value=False); self.rule=tk.StringVar(value='不分類_只匯出選取')
        self.prog=tk.DoubleVar(); self.refs=[]; self.results=[]; self.visible=[]; self.sel=set(); self.stop=False; self.q=queue.Queue(); self.cache=Cache(); self.ui(); self.after(100,self.poll)
    def ui(self):
        bottom=ttk.Frame(self,padding=8); bottom.pack(side='bottom',fill='x'); self.go=ttk.Button(bottom,text='快速掃描預覽',command=self.start); self.go.pack(side='left'); ttk.Button(bottom,text='停止',command=lambda:setattr(self,'stop',True)).pack(side='left',padx=6); ttk.Button(bottom,text='全選目前結果',command=self.selectall).pack(side='left',padx=(12,4)); ttk.Button(bottom,text='清除選取',command=self.clear).pack(side='left'); ttk.Button(bottom,text='輸出選取照片',command=self.export).pack(side='right'); ttk.Button(bottom,text='開啟輸出資料夾',command=self.openout).pack(side='right',padx=6)
        m=ttk.Frame(self); m.pack(fill='both',expand=True); ttk.Label(m,text=APP,font=('Microsoft JhengHei UI',20,'bold')).pack(pady=(9,0)); ttk.Label(m,text='標準模式＋進階模式｜先快速掃描預覽，再確認選取輸出').pack(); ttk.Label(m,text=OWNER,font=('Microsoft JhengHei UI',9,'bold')).pack(pady=(0,5))
        top=ttk.Frame(m); top.pack(fill='x',padx=14)
        for r,(t,v,c) in enumerate([('來源照片：',self.src,self.picksrc),('輸出位置：',self.dst,self.pickdst)]): ttk.Label(top,text=t,width=11).grid(row=r,column=0); ttk.Entry(top,textvariable=v).grid(row=r,column=1,sticky='ew',padx=5,pady=2); ttk.Button(top,text='選擇資料夾',command=c).grid(row=r,column=2)
        top.columnconfigure(1,weight=1)
        f=ttk.LabelFrame(m,text='使用模式'); f.pack(fill='x',padx=14,pady=3); ttk.Radiobutton(f,text='標準模式｜人數・日期・場景・內容標籤',variable=self.mode,value='標準模式',command=self.mod).grid(row=0,column=0,padx=8,pady=5); ttk.Radiobutton(f,text='進階模式｜標準功能＋指定人物高精度比對',variable=self.mode,value='進階模式',command=self.mod).grid(row=0,column=1,padx=8); ttk.Label(f,text='人臉偵測：').grid(row=0,column=2,padx=(20,2)); ttk.Combobox(f,textvariable=self.sens,state='readonly',width=10,values=['精準優先','平衡','團體照優先']).grid(row=0,column=3)
        o=ttk.LabelFrame(m,text='快速掃描內容（未按輸出前，不複製任何照片）'); o.pack(fill='x',padx=14,pady=3); ttk.Checkbutton(o,text='人數辨識',variable=self.people).pack(side='left',padx=8,pady=4); ttk.Checkbutton(o,text='日期',variable=self.date).pack(side='left',padx=8); ttk.Checkbutton(o,text='場景＋內容標籤',variable=self.tags).pack(side='left',padx=8); ttk.Checkbutton(o,text='使用快取（再次掃描更快）',variable=self.usecache).pack(side='left',padx=8)
        r=ttk.LabelFrame(m,text='進階模式｜指定人物參考照片'); r.pack(fill='x',padx=14,pady=3); ttk.Button(r,text='＋ 加入參考照片',command=self.addref).grid(row=0,column=0,padx=8,pady=5); ttk.Button(r,text='清除參考照片',command=self.clearref).grid(row=0,column=1); ttk.Label(r,text='比對嚴格度：').grid(row=0,column=2,padx=(20,3)); ttk.Combobox(r,textvariable=self.strict,state='readonly',width=10,values=['嚴格','標準','寬鬆']).grid(row=0,column=3); self.refl=ttk.Label(r,text='標準模式目前不需參考照片'); self.refl.grid(row=1,column=0,columnspan=4,sticky='w',padx=8,pady=(0,5))
        fl=ttk.LabelFrame(m,text='預覽篩選'); fl.pack(fill='x',padx=14,pady=3); ttk.Label(fl,text='人數').grid(row=0,column=0,padx=(6,2)); ttk.Combobox(fl,textvariable=self.pf,state='readonly',width=12,values=['全部','未偵測到人臉','單人','雙人','3-5人','6-10人','11人以上']).grid(row=0,column=1,pady=4); ttk.Label(fl,text='場景').grid(row=0,column=2,padx=(8,2)); ttk.Combobox(fl,textvariable=self.sf,state='readonly',width=12,values=['全部','室內','人物活動','戶外／自然','街景／交通','餐飲場景','運動場景','其他／待確認']).grid(row=0,column=3); ttk.Label(fl,text='日期含').grid(row=0,column=4,padx=(8,2)); ttk.Entry(fl,textvariable=self.df,width=12).grid(row=0,column=5); ttk.Label(fl,text='標籤／檔名').grid(row=0,column=6,padx=(8,2)); ttk.Entry(fl,textvariable=self.kw,width=15).grid(row=0,column=7); ttk.Checkbutton(fl,text='只看人物候選',variable=self.onlymatch).grid(row=0,column=8,padx=6); ttk.Button(fl,text='套用篩選',command=self.filter).grid(row=0,column=9,padx=5)
        ex=ttk.Frame(m); ex.pack(fill='x',padx=16,pady=2); ttk.Label(ex,text='輸出分類方式：').pack(side='left'); ttk.Combobox(ex,textvariable=self.rule,state='readonly',width=23,values=['不分類_只匯出選取','依人數','依日期_年月','依場景','依指定人物結果']).pack(side='left'); ttk.Label(ex,text='   雙擊預覽列可切換 ✓ 選取').pack(side='left')
        ttk.Progressbar(m,variable=self.prog,maximum=100).pack(fill='x',padx=14,pady=(2,0)); self.status=ttk.Label(m,text='請先選擇來源照片資料夾，再按「快速掃描預覽」。'); self.status.pack(anchor='w',padx=14)
        tf=ttk.Frame(m); tf.pack(fill='both',expand=True,padx=14,pady=4); cols=('sel','name','people','date','scene','match','tags'); self.tree=ttk.Treeview(tf,columns=cols,show='headings');
        for c,t,w in [('sel','選取',45),('name','檔名',230),('people','人數',55),('date','日期',145),('scene','場景',100),('match','人物比對',145),('tags','AI內容標籤',350)]: self.tree.heading(c,text=t); self.tree.column(c,width=w,anchor='center' if c in {'sel','people','date','scene','match'} else 'w')
        y=ttk.Scrollbar(tf,orient='vertical',command=self.tree.yview); x=ttk.Scrollbar(tf,orient='horizontal',command=self.tree.xview); self.tree.configure(yscrollcommand=y.set,xscrollcommand=x.set); self.tree.grid(row=0,column=0,sticky='nsew'); y.grid(row=0,column=1,sticky='ns'); x.grid(row=1,column=0,sticky='ew'); tf.columnconfigure(0,weight=1); tf.rowconfigure(0,weight=1); self.tree.bind('<Double-1>',self.toggle)
    def picksrc(self):
        p=filedialog.askdirectory();
        if p:self.src.set(p); self.dst.set(self.dst.get() or str(Path(p).parent/'AI智慧照片分類結果_v2'))
    def pickdst(self):
        p=filedialog.askdirectory();
        if p:self.dst.set(p)
    def mod(self):
        self.refl.config(text=('已加入 %d 張：'%len(self.refs))+('、'.join(Path(x).name for x in self.refs[:5]) if self.refs else '請加入 2～10 張清楚參考照') if self.mode.get()=='進階模式' else '標準模式目前不需參考照片')
    def addref(self):
        fs=filedialog.askopenfilenames(filetypes=[('照片','*.jpg *.jpeg *.png *.webp *.heic *.heif *.bmp *.tif *.tiff')]); [self.refs.append(f) for f in fs if f not in self.refs]; self.mode.set('進階模式'); self.mod()
    def clearref(self):self.refs=[];self.mod()
    def start(self):
        if not Path(self.src.get()).is_dir():messagebox.showwarning('提示','請先選擇來源照片資料夾。');return
        if self.mode.get()=='進階模式' and not self.refs:messagebox.showwarning('提示','進階模式請先加入指定人物參考照片。');return
        self.stop=False;self.go.config(state='disabled');self.prog.set(0);threading.Thread(target=self.worker,daemon=True).start()
    def worker(self):
        try:
            files=[p for p in Path(self.src.get()).rglob('*') if p.is_file() and p.suffix.lower() in EXTS]; mode=self.mode.get(); rs=refsig(self.refs) if mode=='進階模式' else ''; det=Detector({'精準優先':.68,'平衡':.55,'團體照優先':.42}[self.sens.get()]); tg=Tags() if self.tags.get() else None; mat=None
            if mode=='進階模式':
                mat=ArcMatcher(); n=mat.build(self.refs,det)
                if not n:raise RuntimeError('參考照片未建立到可用的人臉模板，請改用清楚、正面的照片。')
            out=[];hits=0
            for i,p in enumerate(files,1):
                if self.stop:break
                c=self.cache.get(p,mode,rs) if self.usecache.get() else None
                if c:out.append(c);hits+=1
                else:
                    try:
                        im,dt=load_img(p,1600); fs=det.faces(im) if self.people.get() or mat else []; names=tg.run(im) if tg else []; sc=scene(names) if tg else ''; ms=None;ml='';qn=''
                        if mat:
                            ms,qn=mat.score(im,fs); hi,mid={'嚴格':(.58,.48),'標準':(.51,.41),'寬鬆':(.45,.35)}[self.strict.get()]; ml='低品質待確認' if ms is None and fs else ('未偵測人臉' if ms is None else ('高度符合' if ms>=hi else ('疑似符合' if ms>=mid else '不符合')))
                        x={'path':str(p),'name':p.name,'date':dt if self.date.get() else '','people':len(fs) if self.people.get() else 0,'scene':sc,'tags':'、'.join(ZH.get(n,n) for n in names[:10]),'score':ms,'level':ml,'note':qn};out.append(x);self.cache.put(p,mode,rs,x) if self.usecache.get() else None
                    except Exception as e:out.append({'path':str(p),'name':p.name,'date':'','people':0,'scene':'辨識失敗','tags':str(e),'score':None,'level':'辨識失敗','note':''})
                if i%3==0 or i==len(files):self.q.put(('p',100*i/max(1,len(files)),f'快速掃描：{i}/{len(files)}｜快取命中 {hits} 張'))
            self.q.put(('done',out,hits,self.stop))
        except Exception as e:self.q.put(('err',str(e)))
    def okpeople(self,n):
        f=self.pf.get();return f=='全部' or (f=='未偵測到人臉' and n==0) or (f=='單人' and n==1) or (f=='雙人' and n==2) or (f=='3-5人' and 3<=n<=5) or (f=='6-10人' and 6<=n<=10) or (f=='11人以上' and n>=11)
    def filter(self):
        self.tree.delete(*self.tree.get_children());self.visible=[];sf=self.sf.get();df=self.df.get().strip().lower();kw=self.kw.get().strip().lower()
        for i,r in enumerate(self.results):
            if not self.okpeople(int(r.get('people',0))):continue
            if sf!='全部' and r.get('scene')!=sf:continue
            if df and df not in r.get('date','').lower():continue
            if kw and kw not in (r.get('name','')+' '+r.get('tags','')).lower():continue
            if self.onlymatch.get() and r.get('level') not in {'高度符合','疑似符合','低品質待確認'}:continue
            self.visible.append(i);ms=r.get('score');m=r.get('level','')+(f' {ms:.3f}' if ms is not None else '');self.tree.insert('', 'end', iid=str(i), values=('✓' if i in self.sel else '',r.get('name',''),r.get('people',0),r.get('date','')[:16],r.get('scene',''),m,r.get('tags','')))
        self.status.config(text=f'目前顯示 {len(self.visible)} 張｜已選取 {len(self.sel)} 張')
    def toggle(self,e=None):
        iid=self.tree.focus();
        if not iid:return
        i=int(iid); self.sel.remove(i) if i in self.sel else self.sel.add(i); v=list(self.tree.item(iid,'values'));v[0]='✓' if i in self.sel else '';self.tree.item(iid,values=v);self.status.config(text=f'目前顯示 {len(self.visible)} 張｜已選取 {len(self.sel)} 張')
    def selectall(self):self.sel.update(self.visible);self.filter()
    def clear(self):self.sel.clear();self.filter()
    def sub(self,r):
        if self.rule.get()=='依人數':return bucket(int(r.get('people',0)))
        if self.rule.get()=='依日期_年月':return r.get('date','')[:7].replace('-','_') or '日期不明'
        if self.rule.get()=='依場景':return r.get('scene') or '場景不明'
        if self.rule.get()=='依指定人物結果':return r.get('level') or '未做人員比對'
        return '已選取照片'
    def export(self):
        if not self.sel:messagebox.showwarning('提示','請先選取要輸出的照片。');return
        threading.Thread(target=self.expworker,daemon=True).start()
    def expworker(self):
        try:
            base=Path(self.dst.get());base.mkdir(parents=True,exist_ok=True);rows=[];ids=sorted(self.sel)
            for j,i in enumerate(ids,1):
                r=self.results[i];p=Path(r['path'])
                if not p.exists():continue
                d=unique(base/self.sub(r),p);shutil.copy2(p,d);rows.append([str(p),p.name,r.get('date',''),r.get('people',0),r.get('scene',''),r.get('tags',''),r.get('score',''),r.get('level',''),str(d)]);self.q.put(('p',100*j/len(ids),f'正在輸出：{j}/{len(ids)}'))
            cp=base/f'峻爸_AI照片輸出結果_{datetime.now():%Y%m%d_%H%M%S}.csv';f=open(cp,'w',newline='',encoding='utf-8-sig');w=csv.writer(f);w.writerow(['原始路徑','檔名','拍攝日期','偵測人數','場景','AI內容標籤','人物相似度','人物判定','輸出路徑']);w.writerows(rows);f.close();self.q.put(('exp',len(rows),str(cp)))
        except Exception as e:self.q.put(('err',str(e)))
    def openout(self):
        if not self.dst.get():messagebox.showinfo('提示','尚未指定輸出資料夾。');return
        p=Path(self.dst.get());p.mkdir(parents=True,exist_ok=True);os.startfile(p)
    def poll(self):
        try:
            while True:
                x=self.q.get_nowait()
                if x[0]=='p':self.prog.set(x[1]);self.status.config(text=x[2])
                elif x[0]=='done':self.results=x[1];self.sel.clear();self.go.config(state='normal');self.prog.set(100 if not x[3] else 0);self.filter();messagebox.showinfo('快速掃描完成',f'已分析 {len(self.results)} 張照片。\n快取命中 {x[2]} 張。\n目前尚未複製任何照片，請預覽後再選取輸出。')
                elif x[0]=='exp':self.prog.set(100);messagebox.showinfo('輸出完成',f'已輸出 {x[1]} 張選取照片。\nCSV：{x[2]}')
                elif x[0]=='err':self.go.config(state='normal');messagebox.showerror('錯誤',x[1])
        except queue.Empty:pass
        self.after(100,self.poll)
if __name__=='__main__':App().mainloop()
