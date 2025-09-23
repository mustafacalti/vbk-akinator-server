from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Literal
from uuid import uuid4
import math
import random

app = FastAPI(title="YTU-Akinator-Server (Branching)")

# CORS ayarları - Frontend'in erişebilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler yazılmalı
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5 ekip sınıfı ---
CLASSES = [
    "Proje-Yarışma",
    "Medya",
    "Network",
    "Organizasyon",
    "Eğitim",
]

# Likert ölçeği
AnswerKey = Literal[
    "kesinlikle_evet", "evet", "bilmiyorum", "hayir", "kesinlikle_hayir"
]
LIKERT: Dict[AnswerKey, int] = {
    "kesinlikle_evet": 2,
    "evet": 1,
    "bilmiyorum": 0,
    "hayir": -1,
    "kesinlikle_hayir": -2,
}

# Dallanma için cevap kovaları: pozitif / nötr / negatif
def bucket_of(val: int) -> str:
    if val >= 1:
        return "pos"
    if val <= -1:
        return "neg"
    return "neu"

# --- API modelleri ---
class AnswerIn(BaseModel):
    session_id: str
    answer: AnswerKey

class StartOut(BaseModel):
    session_id: str
    question_index: int
    question: str
    choices: List[str]

class NextOut(BaseModel):
    done: bool
    question_index: Optional[int] = None
    question: Optional[str] = None
    choices: Optional[List[str]] = None
    prediction: Optional[str] = None
    confidences: Optional[Dict[str, float]] = None

# --- Parametreler ---
CONFIDENCE_THRESHOLD = 0.75
MAX_QUESTIONS = 8  # Daha kısa quiz
MIN_QUESTIONS = 3  # Minimum soru sayısı
UNCERTAINTY_THRESHOLD = 0.3  # Bu değerin altında belirsiz sayılır (30%)

# --- SORU HAVUZU ---
# Her soru:
#   q: soru metni
#   w: {class: weight} -> puan katkısı
#   category: birincil kategori (başlangıç seçimi için)
#   tags: ek özellikler
QUESTION_POOL: List[Dict] = [
    # Proje-Yarışma soruları (Veri Bilimi, AI, ML odaklı)
    {
        "q": "Veri analizi ve makine öğrenmesi projelerinde çalışmak ilgimi çeker.",
        "w": {"Proje-Yarışma": 1.6, "Eğitim": 0.4},
        "category": "Proje-Yarışma",
        "tags": ["veri_bilimi", "makine_öğrenmesi"]
    },
    {
        "q": "Yapay zeka ve derin öğrenme konularına merak duyarım.",
        "w": {"Proje-Yarışma": 1.5, "Eğitim": 0.5},
        "category": "Proje-Yarışma",
        "tags": ["yapay_zeka", "derin_öğrenme"]
    },
    {
        "q": "Veri setleriyle çalışmak ve anlamlı sonuçlar çıkarmak hoşuma gider.",
        "w": {"Proje-Yarışma": 1.4, "Eğitim": 0.6},
        "category": "Proje-Yarışma",
        "tags": ["veri_analizi", "istatistik"]
    },
    {
        "q": "Algoritma geliştirme ve model eğitimi konularında kendimi geliştirmek isterim.",
        "w": {"Proje-Yarışma": 1.7, "Eğitim": 0.3},
        "category": "Proje-Yarışma",
        "tags": ["algoritma", "model_eğitimi"]
    },
    {
        "q": "Kaggle yarışmaları ve veri bilimi projeleri ilgimi çeker.",
        "w": {"Proje-Yarışma": 1.8, "Network": 0.2},
        "category": "Proje-Yarışma",
        "tags": ["kaggle", "yarışma"]
    },

    # Eğitim soruları
    {
        "q": "Bilgimi başkalarıyla paylaşmak ve öğretmek hoşuma gider.",
        "w": {"Eğitim": 1.7, "Network": 0.3},
        "category": "Eğitim",
        "tags": ["öğretme", "paylaşım"]
    },
    {
        "q": "Workshop ve eğitim etkinlikleri düzenlemek isterim.",
        "w": {"Eğitim": 1.6, "Organizasyon": 0.4},
        "category": "Eğitim",
        "tags": ["workshop", "etkinlik"]
    },
    {
        "q": "Sunum yapmak ve topluluk önünde konuşmak beni heyecanlandırır.",
        "w": {"Eğitim": 1.4, "Medya": 0.3, "Network": 0.3},
        "category": "Eğitim",
        "tags": ["sunum", "konuşma"]
    },
    {
        "q": "Eğitim materyalleri hazırlamak ve kurs içerikleri geliştirmek ilgimi çeker.",
        "w": {"Eğitim": 1.5, "Proje-Yarışma": 0.5},
        "category": "Eğitim",
        "tags": ["materyal", "içerik"]
    },

    # Organizasyon soruları
    {
        "q": "Etkinlik planlaması ve organizasyon işleri beni motive eder.",
        "w": {"Organizasyon": 1.8, "Network": 0.2},
        "category": "Organizasyon",
        "tags": ["planlama", "organizasyon"]
    },
    {
        "q": "Detay odaklı çalışmak ve süreçleri yönetmek hoşuma gider.",
        "w": {"Organizasyon": 1.5, "Proje-Yarışma": 0.5},
        "category": "Organizasyon",
        "tags": ["detay", "süreç"]
    },
    {
        "q": "Stresli durumları yönetmek ve soğukkanlı kalmak güçlü yanlarımdan.",
        "w": {"Organizasyon": 1.4, "Proje-Yarışma": 0.6},
        "category": "Organizasyon",
        "tags": ["stres", "soğukkanlılık"]
    },
    {
        "q": "Liderlik yapmak ve takımları koordine etmek isterim.",
        "w": {"Organizasyon": 1.3, "Network": 0.7},
        "category": "Organizasyon",
        "tags": ["liderlik", "koordinasyon"]
    },

    # Network soruları
    {
        "q": "Yeni insanlarla tanışmak ve ağ kurmak beni mutlu eder.",
        "w": {"Network": 1.6, "Medya": 0.4},
        "category": "Network",
        "tags": ["tanışma", "ağ"]
    },
    {
        "q": "İş birliği ve ortaklık fırsatları aramak ilgimi çeker.",
        "w": {"Network": 1.7, "Organizasyon": 0.3},
        "category": "Network",
        "tags": ["işbirliği", "ortaklık"]
    },
    {
        "q": "Topluluk etkinliklerinde aktif rol almak isterim.",
        "w": {"Network": 1.4, "Eğitim": 0.6},
        "category": "Network",
        "tags": ["topluluk", "aktif_rol"]
    },
    {
        "q": "Dış ilişkiler ve sponsorluk konularında çalışmak hoşuma gider.",
        "w": {"Network": 1.5, "Organizasyon": 0.5},
        "category": "Network",
        "tags": ["dış_ilişkiler", "sponsorluk"]
    },

    # Medya soruları
    {
        "q": "Görsel tasarım ve video içerik üretimi ilgi alanım.",
        "w": {"Medya": 1.8, "Network": 0.3},
        "category": "Medya",
        "tags": ["tasarım", "video"]
    },
    {
        "q": "Sosyal medya platformlarında içerik üretmek ve paylaşmak severim.",
        "w": {"Medya": 1.6, "Network": 0.4},
        "category": "Medya",
        "tags": ["sosyal_medya", "içerik"]
    },
    {
        "q": "Fotoğrafçılık ve görsel hikaye anlatımı ilgimi çeker.",
        "w": {"Medya": 1.5, "Eğitim": 0.5},
        "category": "Medya",
        "tags": ["fotoğraf", "hikaye"]
    },
    {
        "q": "Kreatif yazarlık ve metin içerikleri hazırlamak hoşuma gider.",
        "w": {"Medya": 1.4, "Eğitim": 0.6},
        "category": "Medya",
        "tags": ["yazarlık", "metin"]
    },
    {
        "q": "Kamera karşısında rahatım ve röportaj yapabilirim.",
        "w": {"Medya": 1.3, "Network": 0.7},
        "category": "Medya",
        "tags": ["kamera", "röportaj"]
    }
]

# --- Oturum durumu ---
SESSIONS: Dict[str, Dict] = {}

# --- AKILLI SORU SEÇİMİ ALGORİTMALARI ---

def get_random_starting_question() -> tuple[int, Dict]:
    """Random başlangıç sorusu seç"""
    idx = random.randint(0, len(QUESTION_POOL) - 1)
    return idx, QUESTION_POOL[idx]

def get_next_question(session_state: Dict) -> Optional[tuple[int, Dict]]:
    """Akıllı soru seçimi - çeşitliliği koruyarak"""
    asked_questions = set(session_state.get("asked_questions", []))
    area_counts = session_state.get("area_counts", {area: 0 for area in CLASSES})

    # Henüz sorulmamış soruları filtrele
    available_questions = [
        (i, q) for i, q in enumerate(QUESTION_POOL)
        if i not in asked_questions
    ]

    if not available_questions:
        return None

    # En az sorulan alanı bul
    min_area = min(area_counts, key=area_counts.get)
    min_count = area_counts[min_area]

    # Ağırlıklı seçim: az sorulan alanlara öncelik ver
    weights = []
    for i, q in available_questions:
        category = q["category"]
        count = area_counts[category]
        # En az sorulan alana 3x, diğerlerine 1x ağırlık
        weight = 3.0 if count == min_count else 1.0
        weights.append(weight)

    # Ağırlıklı rastgele seçim
    selected_idx = random.choices(range(len(available_questions)), weights=weights, k=1)[0]
    return available_questions[selected_idx]

def update_session_stats(session_state: Dict, question_idx: int, question: Dict):
    """Oturum istatistiklerini güncelle"""
    if "asked_questions" not in session_state:
        session_state["asked_questions"] = []
    if "area_counts" not in session_state:
        session_state["area_counts"] = {area: 0 for area in CLASSES}
    if "positive_answers" not in session_state:
        session_state["positive_answers"] = 0  # Evet/Kesinlikle evet sayısı

    session_state["asked_questions"].append(question_idx)
    session_state["area_counts"][question["category"]] += 1

def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    mx = max(scores.values()) if scores else 0.0
    exps = {k: math.exp(v - mx) for k, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {k: exps[k] / s for k in scores.keys()}

def should_finish(scores: Dict[str, float], asked: int) -> bool:
    probs = softmax(scores)
    top = max(probs.values())

    # Minimum soru sayısına ulaşmadıysa devam et
    if asked < MIN_QUESTIONS:
        return False

    # Yüksek güvenle tahmin yapabiliyorsa bitir
    if top >= CONFIDENCE_THRESHOLD:
        return True

    # Maximum sorulara ulaştıysa bitir
    if asked >= MAX_QUESTIONS:
        return True

    return False

def is_uncertain_result(scores: Dict[str, float], session_state: Dict) -> bool:
    """Sonuç belirsiz mi kontrol et"""
    probs = softmax(scores)
    top = max(probs.values())

    # Hiç pozitif cevap verilmediyse belirsiz
    if session_state.get("positive_answers", 0) == 0:
        return True

    # Eğer en yüksek skor çok düşükse belirsiz
    if top < UNCERTAINTY_THRESHOLD:
        return True

    # Eğer tüm skorlar çok yakınsa (belirsiz durum)
    score_values = list(probs.values())
    max_score = max(score_values)
    second_max = sorted(score_values, reverse=True)[1]

    # En yüksek ile ikinci en yüksek arasındaki fark çok azsa belirsiz
    if max_score - second_max < 0.05:  # %5 fark
        return True

    return False

@app.get("/")
async def root():
    return {"message": "YTU Akinator Backend is running!", "status": "healthy"}

@app.get("/start", response_model=StartOut)
async def start():
    sid = str(uuid4())

    # Random başlangıç sorusu seç
    question_idx, question = get_random_starting_question()

    SESSIONS[sid] = {
        "i": 0,  # kaç soru soruldu
        "scores": {c: 0.0 for c in CLASSES},
        "asked_questions": [],
        "area_counts": {area: 0 for area in CLASSES},
        "current_question_idx": question_idx,
        "positive_answers": 0
    }

    # İlk soruyu istatistiklere ekle
    update_session_stats(SESSIONS[sid], question_idx, question)

    return StartOut(
        session_id=sid,
        question_index=0,
        question=question["q"],
        choices=list(LIKERT.keys()),
    )

@app.post("/answer", response_model=NextOut)
async def answer(body: AnswerIn):
    if body.session_id not in SESSIONS:
        raise HTTPException(404, "session not found")
    if body.answer not in LIKERT:
        raise HTTPException(400, "invalid answer")

    st = SESSIONS[body.session_id]

    # Mevcut sorunun skorunu güncelle
    current_q_idx = st["current_question_idx"]
    current_question = QUESTION_POOL[current_q_idx]

    val = LIKERT[body.answer]

    # Pozitif cevap sayacını güncelle
    if body.answer in ["evet", "kesinlikle_evet"]:
        st["positive_answers"] = st.get("positive_answers", 0) + 1

    for c in CLASSES:
        st["scores"][c] += current_question.get("w", {}).get(c, 0.0) * val

    # Sayaç
    st["i"] += 1

    # Bitirme kriteri?
    if should_finish(st["scores"], st["i"]):
        return await _finish(st)

    # Sonraki soruyu akıllı seçim ile bul
    next_question_result = get_next_question(st)
    if not next_question_result:
        return await _finish(st)

    next_q_idx, next_question = next_question_result

    # Sonraki soruyu kaydet ve istatistikleri güncelle
    st["current_question_idx"] = next_q_idx
    update_session_stats(st, next_q_idx, next_question)

    return NextOut(
        done=False,
        question_index=st["i"],
        question=next_question["q"],
        choices=list(LIKERT.keys()),
    )

async def _finish(st: Dict) -> NextOut:
    probs = softmax(st["scores"])
    probs = {k: round(v, 4) for k, v in probs.items()}

    # Belirsiz sonuç kontrolü (session state de gönder)
    if is_uncertain_result(st["scores"], st):
        return NextOut(
            done=True,
            prediction="Belirsiz",
            confidences=probs
        )

    pred = max(probs, key=probs.get)
    return NextOut(done=True, prediction=pred, confidences=probs)
