import os
import cv2
import threading
import torch
import numpy as np
from datetime import datetime
from src.realtime_transcriber import RealtimeTranscriber
from src.multi_speaker_verifier import MultiSpeakerVerifier
from src.yolo_detector import YOLOFaceDetector
from src.personid_tracker import PersonIDTracker

class MultimodalFusion:
    def __init__(self, hf_token, device="cuda"):
        self.device = 'cuda' if torch.cuda.is_available() and device == "cuda" else 'cpu'
        self.shared_state = {"active_faces": {}}
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        emb_dir = os.path.join(base_dir, "data", "embeddings")
        
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir, exist_ok=True)
        
        self.verifier = MultiSpeakerVerifier(emb_dir, threshold=0.65)
        self.transcriber = RealtimeTranscriber(
            self.verifier, 
            hf_token, 
            device=self.device,
            verifier_confidence_min=0.9,
            shared_state=self.shared_state
        )
        
        self.detector = YOLOFaceDetector(model_path="od_model/yolov8n-face.pt", device=self.device)
        self.tracker = PersonIDTracker(model_path="od_model/edgeface_xxs.pt", device=self.device)
        self.tracker.load_known_embeddings(emb_dir)

    def video_loop(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Erro: Não foi possível abrir a webcam.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            bboxes = self.detector.detect(frame)
            results = self.tracker.update(frame, bboxes)
            
            current_faces = {}
            for res in results:
                name = res['name']
                track_id = res['track_id']
                current_faces[track_id] = name
                
                x1, y1, x2, y2 = res['bbox']
                label = f"{name} ({res['confidence']:.2f})"
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            self.shared_state["active_faces"] = current_faces
            
            cv2.imshow("AV-Tracker Multimodal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        video_thread = threading.Thread(target=self.video_loop, daemon=True)
        video_thread.start()
        
        print("\n" + "="*60)
        print("🚀 SISTEMA MULTIMODAL ATIVO")
        print("Câmera e Microfone monitorando simultaneamente...")
        print("="*60 + "\n")
        
        self.transcriber.start_recording()

if __name__ == "__main__":
    HF_TOKEN = "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"
    app = MultimodalFusion(HF_TOKEN)
    app.run()