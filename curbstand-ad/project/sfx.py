import numpy as np, wave, sys, os
SR=48000; out=sys.argv[1]; os.makedirs(out,exist_ok=True)
rng=np.random.default_rng(7)
def save(n,x):
    x=x/ (np.max(np.abs(x))+1e-9)*0.9
    s=np.stack([x,x],1); s=(s*32767).astype(np.int16)
    w=wave.open(f"{out}/{n}.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(s.tobytes()); w.close()
def t(d): return np.arange(int(SR*d))/SR
def lp(x,fc):  # time-varying one-pole lowpass
    fc=np.broadcast_to(fc,x.shape); a=np.exp(-2*np.pi*fc/SR); y=np.zeros_like(x); p=0.
    for i in range(len(x)): p=(1-a[i])*x[i]+a[i]*p; y[i]=p
    return y
def env(tt,a,d): return np.minimum(tt/a,1)*np.exp(-np.maximum(tt-a,0)/d)
# message send: quick upward blip
tt=t(.18); f=600+1400*tt/.18; save("send", np.sin(2*np.pi*np.cumsum(f)/SR)*env(tt,.004,.05))
# receive: two-tone glass
tt=t(.45); x=sum(np.sin(2*np.pi*fr*tt)*env(tt,.003,.12)*g for fr,g in [(1318,1),(2637,.3)]); x[int(.09*SR):]+=(np.sin(2*np.pi*1760*tt)*env(tt,.003,.15))[:len(tt)-int(.09*SR)]; save("receive",x)
# typing ticks
tt=t(.6); x=np.zeros_like(tt)
for k in range(7):
    s=int((k*.08+.01*(k%3))*SR); n=int(.012*SR); x[s:s+n]+=rng.standard_normal(n)*np.exp(-np.arange(n)/(.002*SR))*.6
save("typing",lp(x,np.full_like(x,5000)))
# whoosh
tt=t(.7); n=rng.standard_normal(len(tt)); fc=300+5000*np.sin(np.pi*tt/.7)**2; save("whoosh", lp(n,fc)*np.sin(np.pi*tt/.7)**1.5)
# thud (card land)
tt=t(.35); f=140*np.exp(-tt*9)+55; x=np.sin(2*np.pi*np.cumsum(f)/SR)*env(tt,.002,.09)+lp(rng.standard_normal(len(tt)),np.full(len(tt),1800.))*env(tt,.001,.02)*.6; save("thud",x)
# pin drop: click + pitched pluck
tt=t(.4); x=np.sin(2*np.pi*(880)*tt)*env(tt,.002,.07)+np.sin(2*np.pi*1320*tt)*env(tt,.002,.04)*.4; x[:200]+=rng.standard_normal(200)*.8; save("pin",x)
# riser
tt=t(1.6); n=rng.standard_normal(len(tt)); fc=200+7000*(tt/1.6)**2; x=lp(n,fc)*(tt/1.6)**2+np.sin(2*np.pi*np.cumsum(200+600*(tt/1.6)**2)/SR)*(tt/1.6)**3*.3; save("riser",x)
# logo chime: chord with shimmer
tt=t(2.5); x=sum(np.sin(2*np.pi*fr*tt+np.sin(2*np.pi*5*tt)*.2)*env(tt,.01,.9)*g for fr,g in [(523.25,1),(659.25,.8),(783.99,.7),(1046.5,.4),(1567.98,.15)])
x+=np.sin(2*np.pi*65*tt)*env(tt,.005,.3)*1.2; save("chime",x)
# soft music bed: pulsing pad 22s, Am-F-C-G
D=22; tt=t(D); x=np.zeros_like(tt)
chords=[[220,261.63,329.63],[174.61,220,261.63],[261.63,329.63,392],[196,246.94,293.66]]
for i in range(int(D/2.75)+1):
    c=chords[i%4]; s=int(i*2.75*SR); e=min(len(tt),s+int(3.2*SR)); seg=tt[:e-s]
    for fr in c:
        x[s:e]+=(np.sin(2*np.pi*fr*seg)+.3*np.sin(2*np.pi*fr*2*seg))*np.minimum(seg/.4,1)*np.exp(-seg/2.2)*.25
    # kick-ish pulse on beats
    for b in range(4):
        bs=s+int(b*.6875*SR); bt=tt[:int(.25*SR)]
        if bs+len(bt)<len(tt): x[bs:bs+len(bt)]+=np.sin(2*np.pi*np.cumsum(90*np.exp(-bt*20)+45)/SR)*np.exp(-bt*14)*.5
save("bed",x*np.minimum(1,np.minimum(tt/1.0,(D-tt)/1.5)))
