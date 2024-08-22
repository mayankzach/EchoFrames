from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream
import assemblyai as aai
aai.settings.api_key = "enter key"
translation_template = """
Translate the following sentence into {language}, return ONLY the translation, nothing else.

Sentence: {sentence}"""


output_parser = StrOutputParser()
llm = ChatOpenAI(api_key="enter key",temperature=0.0, model="gpt-4-turbo")
translation_prompt = ChatPromptTemplate.from_template(translation_template)

translation_chain = (
    {"language": RunnablePassthrough(), "sentence": RunnablePassthrough()} 
    | translation_prompt
    | llm
    | output_parser
)

def translate(sentence, language="French"):
    print("now gpt")
    data_input = {"language": language, "sentence": sentence}
    translation = translation_chain.invoke(data_input)
    print("end gpt")
    return translation


client = ElevenLabs(
    api_key="enter key",
)
# client.generate(
#         text=text,
#         voice="pNInz6obpgDQGcFmaJgB", # Insert voice model here!
#         model="eleven_multilingual_v2"
#     )
def gen_dub(text):
    print("Generating audio...")
    audio=client.text_to_speech.convert_as_stream(
    voice_id="pMsXgVXv3BLzUgSXRplE",
    optimize_streaming_latency="0",
    output_format="mp3_22050_32",
    text=text,
    voice_settings=VoiceSettings(
        stability=0.1,
        similarity_boost=0.3,
        style=0.2,
    ),)
    play(audio)

def on_open(session_opened: aai.RealtimeSessionOpened):
  "This function is called when the connection has been established."
  print("Session ID:", session_opened.session_id)

def on_data(transcript: aai.RealtimeTranscript):
  "This function is called when a new transcript has been received."
  if not transcript.text:
    return

  if isinstance(transcript, aai.RealtimeFinalTranscript):
    print(transcript.text, end="\r\n")
    print("Translating...")
    translation = translate(str(transcript.text))
    print(f"Translation: {translation}")
    gen_dub(translation)
  else:
    print(transcript.text, end="\r")
      
def on_error(error: aai.RealtimeError):
  "This function is called when the connection has been closed."
  print("An error occured:", error)

def on_close():
  "This function is called when the connection has been closed."
  print("Closing Session")

# transcriber = aai.RealtimeTranscriber(
#   on_data=on_data,
#   on_error=on_error,
#   sample_rate=44_100,
#   on_open=on_open, # optional
#   on_close=on_close, # optional
# )

# # Start the connection, likely have to restart kernal (runs better as full code in something like VSCode)
# transcriber.connect()
# microphone_stream = aai.extras.MicrophoneStream()
# transcriber.stream(microphone_stream)
# URL of the file to transcribe
FILE_URL = "https://github.com/AssemblyAI-Community/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

# def transcribe_audio(filename):
#     # Open the audio file for reading
#     with open(filename, "rb") as audio_file:
#         print("Transcribing audio...")
#         transcript = openai.Audio.transcribe("whisper-1", audio_file)
#         print("Transcription completed.")
#         return transcript['text']

import requests
import time

# AssemblyAI API key
API_KEY = 'enter key'

# Step 1: Post audio file to AssemblyAI and get transcript ID
def post_audio_and_get_transcript_id(audio_url):
    print("Posting audio file to AssemblyAI...")
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "audio_url": audio_url,
            "auto_chapters": False,
            "auto_highlights": False,
            "boost_param": "high",
            "dual_channel": False,
            "format_text": True,
            "language_detection": True,
            "punctuate": True
        },
    )

    if response.status_code == 200:
        transcript_id = response.json()["id"]
        print(f"Transcript ID: {transcript_id}")
        return transcript_id
    else:
        print(f"Failed to post audio. Status code: {response.status_code}")
        print(response.json())
        return None

# Step 2: Get the transcription text using the transcript ID
def get_transcription_text(transcript_id):
    print(f"Getting transcription for ID: {transcript_id}...")
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    
    while True:
        response = requests.get(url, headers={"Authorization": API_KEY})
        status = response.json().get("status")

        if status == "completed":
            print("Transcription completed.")
            return response.json()["text"]
        elif status == "failed":
            print("Transcription failed.")
            return None
        else:
            print(f"Status: {status}. Waiting for transcription to complete...")
            time.sleep(5)  # Wait for 5 seconds before retryin


if __name__ == "__main__":
    audio_url = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3" 
    # audio_file_path = "C:/Users/mayan/Downloads/WhatsApp Audio 2024-08-22 at 6.32.34 PM.aac"
        # Upload a media file (POST /v2/upload)
    # response = requests.post(
    #     "https://api.assemblyai.com/v2/upload",
    #     headers={
    #         "Authorization": "f337f6858e7649619cb85bfa496f4f98",
    #         "Content-Type": "application/octet-stream"
    #     },
    #     data=open(audio_file_path, 'rb').read(),
    # )
    # audio_url=response.json()['upload_url']
    print("STT")
    transcript_id = post_audio_and_get_transcript_id(audio_url)
    
    if transcript_id:
        transcript_text = get_transcription_text(transcript_id)
        print(transcript_text)
        print("End STT")

# Path to your audio file
        



# transcriber = aai.Transcriber()
# print("now STT")
# transcript = transcriber.transcribe(FILE_URL)

# if transcript.status == aai.TranscriptStatus.error:
#     print(transcript.error)
# else:
#     print("End STT")
#     gen_dub(translate(str(transcript.text[100:160])))