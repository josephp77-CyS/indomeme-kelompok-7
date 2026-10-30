"""Render the implemented INDOMEME architecture and reuse approved figures byte for byte."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

OUT=ROOT/'docs/figures'
OLD=OUT
OUT.mkdir(parents=True,exist_ok=True)
reused=[]
for name in ['alur_data_dan_eksperimen','jumlah_prediksi_benar','skor_seimbang_macro_f1']:
    for suffix in ['.png','.svg']:
        source=OLD/(name+suffix)
        target=OUT/source.name
        if source.resolve()!=target.resolve():shutil.copyfile(source,target)
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
        reused.append({'file':target.name,'sha256':sha,'unchanged':True})

SOURCE=ROOT/'src/master_training.py'
code=SOURCE.read_text(encoding='utf-8')
assert "needs_projection=config['kind']=='multimodal' or config['task']=='both'" in code
assert "nn.Sequential(nn.Linear(dim,dim//2),nn.Dropout(config['dropout']))" in code
assert "features.append(text.pooler_output)" in code and "image.pooler_output" in code
assert "torch.cat(features,dim=1)" in code
assert "nn.Linear(out_dim,2)" in code
assert not any(activation in code for activation in ['nn.ReLU','nn.GELU','nn.Tanh'])
for identifier in ['E059','E060','E101','E110']:
    config=json.loads((ROOT/f'results/runs/{identifier}/config.json').read_text())
    assert config['dropout']==.3 and config['lambda_hate']==.5
    assert config['models']['text']['revision']=='32e28c05b47e33b6675d2670a1745c50a65e987a'

NAVY,TEAL,PALE='#142B46','#007F82','#F2F8F8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':20,'text.color':NAVY,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(12.8,8.8))
fig.subplots_adjust(left=.02,right=.98,top=.84,bottom=.02)
fig.suptitle('Arsitektur model yang dijalankan',x=.04,y=.98,ha='left',fontsize=29,weight='bold')
fig.text(.04,.89,'Diagram utama menunjukkan gabungan gambar dan teks diperkaya.',fontsize=21,va='top')
ax.axis('off')
box_items=[]


def box(x,y,w,h,text,size=20):
    patch=FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=0.008,rounding_size=0.005',edgecolor=TEAL,
        facecolor=PALE,linewidth=1.4,transform=ax.transAxes)
    ax.add_patch(patch)
    label=ax.text(x,y,text,fontsize=size,ha='center',va='center',linespacing=1.15,transform=ax.transAxes)
    box_items.append((patch,label))


def arrow(points):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=TEAL,linewidth=1.6,transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
        color=TEAL,linewidth=1.6,mutation_scale=16,transform=ax.transAxes))


box(.12,.79,.20,.18,'Gambar meme\nRGB 224 × 224',21)
box(.12,.50,.20,.20,'Teks rilis:\ntopik anotasi,\ncaption, OCR',20)
box(.37,.79,.20,.18,'ViT\nVektor gambar\n768 angka',21)
box(.37,.50,.20,.20,'IndoBERTweet\nVektor teks\n768 angka',21)
box(.63,.645,.19,.30,'Satukan vektor\n1536 angka\nLinear\n1536 → 768\nDropout 0,3',20)
box(.885,.79,.21,.22,'Satu tugas\n1 head 768 → 2\nKebencian atau\nketidakpantasan',20)
box(.885,.50,.21,.22,'Dua tugas\n2 head 768 → 2\nKebencian dan\nketidakpantasan',20)
ax.text(.885,.955,'Pilihan keluaran',fontsize=20,ha='center',va='center',weight='bold',transform=ax.transAxes)
ax.text(.885,.645,'atau',fontsize=20,ha='center',va='center',transform=ax.transAxes)
arrow([(.228,.79),(.263,.79)])
arrow([(.228,.50),(.263,.50)])
arrow([(.478,.79),(.495,.79),(.495,.645),(.528,.645)])
arrow([(.478,.50),(.495,.50),(.495,.645),(.528,.645)])
arrow([(.733,.645),(.75,.645),(.75,.79),(.772,.79)])
arrow([(.733,.645),(.75,.645),(.75,.50),(.772,.50)])

ax.text(.02,.335,'Varian tanpa gabungan dua encoder',fontsize=22,weight='bold',ha='left',va='top',transform=ax.transAxes)
ax.text(.02,.255,'Gambar atau teks, satu tugas',fontsize=21,weight='bold',ha='left',va='top',transform=ax.transAxes)
ax.text(.02,.193,'Vektor 768 → Dropout 0,3\n→ satu head Linear 768 → 2',fontsize=20,ha='left',va='top',linespacing=1.35,transform=ax.transAxes)
ax.text(.53,.255,'Teks saja, dua tugas',fontsize=21,weight='bold',ha='left',va='top',transform=ax.transAxes)
ax.text(.53,.193,'Vektor 768 → Dropout 0,3\n→ Linear 768 → 384 → Dropout 0,3\n→ dua head Linear 384 → 2',fontsize=20,ha='left',va='top',linespacing=1.35,transform=ax.transAxes)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
width,height=fig.canvas.get_width_height()
for artist in fig.findobj(match=matplotlib.text.Text):
    if artist.get_visible() and artist.get_text().strip():
        b=artist.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=width+1 and b.y1<=height+1,(artist.get_text(),b.bounds)
assert not fig.texts[0].get_window_extent(renderer).overlaps(fig.texts[1].get_window_extent(renderer))
for patch,label in box_items:
    b=label.get_window_extent(renderer)
    frame=patch.get_window_extent(renderer)
    assert b.x0>frame.x0+2 and b.y0>frame.y0+2 and b.x1<frame.x1-2 and b.y1<frame.y1-2,(label.get_text(),b.bounds,frame.bounds)
fig.savefig(OUT/'arsitektur_model.png',dpi=300,facecolor='white')
fig.savefig(OUT/'arsitektur_model.svg',facecolor='white')
plt.close(fig)

caption=('Arsitektur aktual model pada eksperimen inti. Diagram utama memperlihatkan model gabungan. '
    'ViT dan IndoBERTweet masing-masing menghasilkan vektor ringkasan pooler_output sepanjang 768 angka. '
    'Concatenation menyatukan keduanya menjadi 1.536 angka. Proyeksi Linear mengecilkan vektor menjadi 768 angka, '
    'diikuti Dropout 0,3, lalu satu atau dua head klasifikasi. Pilihan satu dan dua tugas adalah konfigurasi run '
    'yang berbeda, bukan empat head yang berjalan bersamaan. Setiap head Linear menghasilkan dua logit '
    'untuk dua kelas pada satu tugas. Kedua encoder ikut dilatih lanjut. Tidak ada ReLU/GELU di antara proyeksi '
    'bersama dan head. Varian gambar/teks satu tugas memakai Dropout lalu satu head tanpa proyeksi bersama. '
    'Varian teks dua tugas memakai Dropout sebelum proyeksi 768 menjadi 384, serta Dropout kedua sesudah proyeksi '
    'dan dua head 384 menjadi 2. Teks diperkaya berasal dari topik anotasi, caption, dan OCR rilis; urutan input '
    'aktualnya topic [SEP] caption [SEP] OCR, dengan panjang maksimum 256 token. Fungsi loss multitask '
    'berbobot 0,5 untuk tiap tugas dijelaskan pada teks; aturan hubungan label diterapkan pada prediksi, '
    'bukan sebagai penalti tambahan pada loss.')
manifest={'file':'arsitektur_model.png','size_inches':[12.8,8.8],'png_dpi':300,
    'width_doc_inches':6.4,'height_doc_inches':4.4,'minimum_visible_font_points_at_doc_width':10,
    'caption':caption,'source_code':str(SOURCE.relative_to(ROOT)),
    'source_code_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'source_functions':['MemeClassifier.__init__','MemeClassifier.forward','total_loss'],
    'reviewer_confirmation':'research_evidence confirmed hidden_size=768 in pinned cached model configs and all branch/dropout details.',
    'encoder_revisions':{'ViT':'98cb6f7099d372694c8547bf7002289b36857dc7','IndoBERTweet':'32e28c05b47e33b6675d2670a1745c50a65e987a'},
    'assertions':{'all_text_inside_canvas':True,'all_box_labels_inside_boxes':True,'title_subtitle_nonoverlapping':True,'source_code_patterns_match':True},
    'reused_figures':reused,
    'files':[{'name':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
        for f in sorted(OUT.iterdir()) if f.suffix in ['.png','.svg']]}
(OUT/'facts_architecture.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'caption_arsitektur.md').write_text('# Caption arsitektur\n\n'+caption+'\n\nSumber: notebooks/master_training.py dan konfigurasi encoder/run yang dipatok.\n',encoding='utf-8')
print(json.dumps({'output':str(OUT),'copied_unchanged':len(reused),'new_figure':'arsitektur_model.png'},ensure_ascii=False))
