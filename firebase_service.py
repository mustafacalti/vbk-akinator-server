import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, List, Optional
import os
from datetime import datetime, timezone

class FirebaseService:
    def __init__(self):
        # Firebase Admin SDK başlatma
        if not firebase_admin._apps:
            # Production'da environment variable'dan key al
            firebase_key_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if firebase_key_json:
                import json
                service_account_info = json.loads(firebase_key_json)
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred)
            elif os.path.exists("firebase-service-key.json"):
                # Development için local file
                cred = credentials.Certificate("firebase-service-key.json")
                firebase_admin.initialize_app(cred)
            else:
                # Default credentials
                firebase_admin.initialize_app()

        self.db = firestore.client()

    async def save_game_result(self, predicted_class: str,
                              asked_questions: List[int], confidences: Dict[str, float],
                              session_data: Dict) -> str:
        """Oyun sonucunu Firebase'e kaydet (anonim)"""
        try:
            print(f"🔥 Firebase save attempt - Class: {predicted_class}")

            game_result = {
                "predicted_class": predicted_class,
                "asked_questions": asked_questions,
                "confidences": confidences,
                "positive_answers": session_data.get("positive_answers", 0),
                "total_questions": len(asked_questions),
                "area_counts": session_data.get("area_counts", {}),
                "timestamp": datetime.now(timezone.utc),
                "is_uncertain": predicted_class == "Belirsiz"
            }

            print(f"🔥 Saving to Firestore: {game_result}")

            # Firestore'a kaydet
            doc_ref = self.db.collection("game_results").add(game_result)
            print(f"✅ Firebase save success! Doc ID: {doc_ref[1].id}")
            return doc_ref[1].id

        except Exception as e:
            print(f"❌ Firebase save error: {e}")
            import traceback
            traceback.print_exc()
            return None


    async def get_global_area_statistics(self) -> Dict[str, float]:
        """Tüm oyunların alan istatistiklerini getir (genel trend)"""
        try:
            # Son 200 oyunu al
            query = (self.db.collection("game_results")
                    .order_by("timestamp", direction=firestore.Query.DESCENDING)
                    .limit(200))

            docs = query.stream()
            games = [doc.to_dict() for doc in docs]

            if not games:
                return {}

            area_counts = {}
            total_games = len([g for g in games if g.get("predicted_class") != "Belirsiz"])

            for game in games:
                predicted_class = game.get("predicted_class")
                if predicted_class and predicted_class != "Belirsiz":
                    area_counts[predicted_class] = area_counts.get(predicted_class, 0) + 1

            # Yüzdelik oranları hesapla
            area_percentages = {}
            for area, count in area_counts.items():
                area_percentages[area] = count / total_games if total_games > 0 else 0

            return area_percentages

        except Exception as e:
            print(f"Firebase get global stats error: {e}")
            return {}

    async def calculate_balanced_area_weights(self) -> Dict[str, float]:
        """Global istatistiklere göre dengeli alan ağırlıklarını hesapla"""
        try:
            global_stats = await self.get_global_area_statistics()

            if not global_stats:
                # Veri yoksa eşit ağırlık
                return {"Proje-Yarışma": 1.0, "Medya": 1.0, "Network": 1.0,
                       "Organizasyon": 1.0, "Eğitim": 1.0}

            # Az çıkan alanların sorularının daha çok sorulması için ağırlık artır
            weights = {}
            for area in ["Proje-Yarışma", "Medya", "Network", "Organizasyon", "Eğitim"]:
                percentage = global_stats.get(area, 0.2)  # Default %20 (eşit dağılım)

                # Az çıkan alanlara daha çok şans ver
                if percentage < 0.15:  # %15'ten az çıkmışsa
                    weights[area] = 1.5  # %50 daha fazla şans
                elif percentage < 0.18:  # %18'den az çıkmışsa
                    weights[area] = 1.3  # %30 daha fazla şans
                elif percentage > 0.25:  # %25'ten fazla çıkmışsa
                    weights[area] = 0.8  # %20 daha az şans
                else:  # Normal aralıkta (%18-25 arası)
                    weights[area] = 1.0  # Normal

            return weights

        except Exception as e:
            print(f"Firebase calculate weights error: {e}")
            # Hata durumunda eşit ağırlık döndür
            return {"Proje-Yarışma": 1.0, "Medya": 1.0, "Network": 1.0,
                   "Organizasyon": 1.0, "Eğitim": 1.0}

# Singleton instance
firebase_service = FirebaseService()