import whisper
import speech_recognition as sr
import numpy as np
import cv2
import asyncio
from threading import Thread
from io import BytesIO
import wave


def recognize_speech_from_mic(model, recognizer, microphone):
  with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio_data = recognizer.listen(source)
    audio_bytes = audio_data.get_wav_data()
  audio_np = convert_wav_to_numpy(audio_bytes)
  result = model.transcribe(audio_np)
  return result["text"]


def convert_wav_to_numpy(audio_bytes):
  with wave.open(BytesIO(audio_bytes), 'rb') as wave_file:
    framerate = wave_file.getframerate()
    nframes = wave_file.getnframes()
    wav_data = wave_file.readframes(nframes)
    audio_np = np.frombuffer(wav_data, dtype=np.int16).astype(
        np.float32) / 32768.0
  return audio_np


class SpeechApp:

  def __init__(self, camera_index=1):
    self.recognizer = sr.Recognizer()
    self.microphone = sr.Microphone()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.camera_index = camera_index
    self.recognized_text = ""
    self.whisper_model = whisper.load_model("base")
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
      text = recognize_speech_from_mic(self.whisper_model, self.recognizer,
                                       self.microphone)
      if text:
        print(f"Said: {text}")
        self.recognized_text = f"Said: {text}"
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
          text_size = cv2.getTextSize(self.recognized_text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
          text_x = (frame.shape[1] - text_size[0]) // 2
          text_y = frame.shape[0] - 10
          cv2.putText(frame, self.recognized_text, (text_x, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
