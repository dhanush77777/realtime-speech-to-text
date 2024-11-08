import pyaudio
import time
from google.cloud import speech
import queue
RATE = 16000
CHUNK = int(RATE / 10)  
audio_queue = queue.Queue()

client = speech.SpeechClient()

streaming_config = speech.StreamingRecognitionConfig(
    config=speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    ),
    interim_results=True  
)

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self.rate, input=True, frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
        self._closed = False

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        if in_data:
            audio_queue.put(in_data)
        return None, pyaudio.paContinue

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._audio_interface.terminate()
        self._closed = True

    def generator(self):
        while not self._closed:
            chunk = audio_queue.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = audio_queue.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses):
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]

        if result.is_final:
            print("\r" + " " * 80, end="")  
            print("\r" + result.alternatives[0].transcript)
            break  
        else:
            print("\r" + result.alternatives[0].transcript, end="", flush=True)


def main():
    print("Listening...")

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        listen_print_loop(responses)


if __name__ == "__main__":
    main()
