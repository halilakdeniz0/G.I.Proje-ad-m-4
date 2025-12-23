# =============================================================
# 🚗 ARAÇ HASAR ANALİZ SİSTEMİ - KISA VERSİYON
# =============================================================

# !pip install gradio matplotlib ultralytics -q

from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import io

# Model
model = YOLO('/content/car_damage_final/training/weights/best.pt')

# Rapor geçmişi
report_history = []

def analyze(image, conf=0.1):
    if image is None:
        return None, "❌ Resim yükleyin!", None
    
    img = np.array(image)
    results = model.predict(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conf=conf, verbose=False)
    annotated = cv2.cvtColor(results[0].plot(line_width=3), cv2.COLOR_BGR2RGB)
    detections = len(results[0].boxes)
    
    if detections == 0:
        return Image.fromarray(annotated), "⚠️ Hasar bulunamadı.", None
    
    damages = []
    total = 0
    for i, box in enumerate(results[0].boxes):
        c = float(box.conf[0])
        cost = int(5000 + 15000 * c)
        total += cost
        damages.append({'no': i+1, 'conf': c, 'cost': cost})
    
    risk = "🔴 YÜKSEK" if total > 50000 else "🟠 ORTA" if total > 25000 else "🟢 DÜŞÜK"
    
    # Raporu kaydet
    report = {
        'date': datetime.now().strftime('%d.%m.%Y %H:%M'),
        'id': f"RPT-{len(report_history)+1:03d}",
        'count': detections,
        'damages': damages,
        'total': total,
        'risk': risk
    }
    report_history.append(report)
    
    # Özet
    summary = f"🔍 {detections} hasar | 💰 ₺{total:,} | {risk}"
    
    # Renkli grafik
    chart = create_chart(damages)
    
    return Image.fromarray(annotated), summary, chart


def create_chart(damages):
    """Renkli istatistik grafiği"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    labels = [f"H{d['no']}" for d in damages]
    costs = [d['cost'] for d in damages]
    confs = [d['conf']*100 for d in damages]
    
    # Bar
    ax1.bar(labels, costs, color=colors[:len(damages)])
    ax1.set_title('Maliyet (₺)', fontweight='bold')
    ax1.set_ylabel('₺')
    
    # Pie
    ax2.pie(confs, labels=labels, colors=colors[:len(damages)], autopct='%1.0f%%')
    ax2.set_title('Güvenilirlik', fontweight='bold')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def show_reports():
    """Tüm rapor geçmişini göster"""
    if not report_history:
        return "📋 Henüz rapor yok."
    
    text = "╔══════════════════════════════════════════════════════╗\n"
    text += "║            📋 RAPOR GEÇMİŞİ                          ║\n"
    text += "╠══════════════════════════════════════════════════════╣\n"
    
    for r in reversed(report_history):
        text += f"║ {r['id']} | {r['date']}\n"
        text += f"║   🔍 {r['count']} hasar | 💰 ₺{r['total']:,} | {r['risk']}\n"
        for d in r['damages']:
            text += f"║   • Hasar {d['no']}: %{d['conf']*100:.0f} - ₺{d['cost']:,}\n"
        text += "╠══════════════════════════════════════════════════════╣\n"
    
    text += "╚══════════════════════════════════════════════════════╝"
    return text


# ===========================================
# GRADIO
# ===========================================
css = """
.gradio-container { background: linear-gradient(135deg, #1a1a1a, #2d2d2d) !important; }
.gr-button-primary { background: #444 !important; }
"""

with gr.Blocks(title="Hasar Analizi", css=css) as demo:
    
    gr.Markdown("# 🚗 ARAÇ HASAR ANALİZ SİSTEMİ")
    
    with gr.Tabs():
        # TAB 1: Analiz
        with gr.Tab("🔍 Analiz"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label="Araç Resmi", height=300)
                    conf = gr.Slider(0.01, 0.5, 0.1, label="Güvenilirlik Eşiği")
                    btn = gr.Button("🔍 ANALİZ ET", variant="primary", size="lg")
                with gr.Column():
                    img_out = gr.Image(label="Sonuç", height=300)
                    summary = gr.Textbox(label="Özet", lines=2)
            
            gr.Markdown("### 📈 İstatistikler")
            chart = gr.Image(label="", height=200)
            
            btn.click(analyze, [img_in, conf], [img_out, summary, chart])
        
        # TAB 2: Raporlar
        with gr.Tab("📋 Raporlar"):
            gr.Markdown("### � Tüm Analiz Geçmişi")
            report_btn = gr.Button("� Raporları Göster", variant="primary")
            report_text = gr.Textbox(label="", lines=25, show_copy_button=True)
            
            report_btn.click(show_reports, [], report_text)
    
    gr.Markdown("---\n⚙️ YOLOv8 | 4105 Görüntü ile eğitildi")

demo.launch(share=True)
