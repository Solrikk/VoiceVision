import speech_recognition as sr
import cv2
import asyncio
from threading import Thread
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_text_on_image(image, text, position, font_path='arial.ttf', font_size=32, color=(0, 255, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def recognize_speech_from_mic(recognizer, microphone):

    if not isinstance(recognizer, sr.Recognizer):
        raise ValueError("`recognizer` must be `speech_recognition.Recognizer` instance")
    if not isinstance(microphone, sr.Microphone):
        raise ValueError("`microphone` must be `speech_recognition.Microphone` instance")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio, language="ru-RU")
    except sr.RequestError:
        return None
    except sr.UnknownValueError:
        return None

class SpeechApp:

    def __init__(self, camera_index=1):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.camera_index = camera_index
        self.recognized_text = ""

        self.start_recognition_thread()
        self.start_video_thread()
        print("Initialization complete")

    def start_recognition_thread(self):
        def run():
            asyncio.run(self.speech_recognition_loop())

        thread = Thread(target=run, daemon=True)
        thread.start()
        print("Recognition thread started")

    def start_video_thread(self):
        def run():
            asyncio.run(self.video_capture_loop(self.camera_index))

        thread = Thread(target=run, daemon=True)
        thread.start()
        print("Video thread started")

    async def speech_recognition_loop(self):
        print("Starting speech recognition loop")
        while True:
            text = recognize_speech_from_mic(self.recognizer, self.microphone)
            if text:
                print(f"Said: {text}")
                self.recognized_text = f"Said: {text}"
            else:
                self.recognized_text = "" 
            await asyncio.sleep(0.1)

    async def video_capture_loop(self, camera_index):
        print("Starting video capture loop")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Failed to open camera with index {camera_index}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                if self.recognized_text:
                    text_size = cv2.getTextSize(self.recognized_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] - 50 
                    frame = draw_text_on_image(frame, self.recognized_text, (text_x, text_y))

                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting application")
    app = SpeechApp(camera_index=1)
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Application stopped manually")
