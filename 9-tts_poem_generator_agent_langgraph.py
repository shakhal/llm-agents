# Import required libraries
from typing import TypedDict
from langgraph.graph import StateGraph, END
from IPython.display import display, Audio, Markdown
from openai import OpenAI
from dotenv import load_dotenv
import io
import tempfile
import re
import os

# Load environment variables and set OpenAI API key
load_dotenv()
# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

## Initialize OpenAI client and define state

client = OpenAI()

## State Definition
class AgentState(TypedDict):
    input_text: str
    processed_text: str
    audio_data: bytes
    audio_path: str
    content_type: str

## Define Node Functions

def classify_content(state: AgentState) -> AgentState:
    """Classify the input text into one of four categories: general, poem, news, or joke."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the content as one of: 'general', 'poem', 'news', 'joke'."},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["content_type"] = response.choices[0].message.content.strip().lower()
    return state

def process_general(state: AgentState) -> AgentState:
    """Process general content (no specific processing, return as-is)."""
    state["processed_text"] = state["input_text"]
    return state

def process_poem(state: AgentState) -> AgentState:
    """Process the input text as a poem, rewriting it in a poetic style."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rewrite the following text as a short, beautiful poem:"},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["processed_text"] = response.choices[0].message.content.strip()
    return state

def process_news(state: AgentState) -> AgentState:
    """Process the input text as news, rewriting it in a formal news anchor style."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rewrite the following text in a formal news anchor style:"},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["processed_text"] = response.choices[0].message.content.strip()
    return state

def process_joke(state: AgentState) -> AgentState:
    """Process the input text as a joke, turning it into a short, funny joke."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Turn the following text into a short, funny joke:"},
            {"role": "user", "content": state["input_text"]}
        ]
    )
    state["processed_text"] = response.choices[0].message.content.strip()
    return state



def text_to_speech(state: AgentState, save_file: bool = False) -> AgentState:
    """
    Converts processed text into speech using a voice mapped to the content type.
    Optionally saves the audio to a file.

    Args:
        state (AgentState): Dictionary containing the processed text and content type.
        save_file (bool, optional): If True, saves the audio to a file. Defaults to False.

    Returns:
        AgentState: Updated state with audio data and file path (if saved).
    """
    
    # Map content type to a voice, defaulting to "alloy"
    voice_map = {
        "general": "alloy",
        "poem": "nova",
        "news": "onyx",
        "joke": "shimmer"
    }
    voice = voice_map.get(state["content_type"], "alloy")
    
    audio_data = io.BytesIO()

    # Generate speech and stream audio data into memory
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=state["processed_text"]
    ) as response:
        for chunk in response.iter_bytes():
            audio_data.write(chunk)
    
    state["audio_data"] = audio_data.getvalue()
    
    # Save audio to a file if requested
    if save_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(state["audio_data"])
            state["audio_path"] = temp_audio.name
    else:
        state["audio_path"] = ""
    
    return state 

## Define and Compile the Graph

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("classify_content", classify_content)
workflow.add_node("process_general", process_general)
workflow.add_node("process_poem", process_poem)
workflow.add_node("process_news", process_news)
workflow.add_node("process_joke", process_joke)
workflow.add_node("text_to_speech", text_to_speech)

# Set the entry point of the graph
workflow.set_entry_point("classify_content")

# Define conditional edges based on content type
workflow.add_conditional_edges(
    "classify_content",
    lambda x: x["content_type"],
    {
        "general": "process_general",
        "poem": "process_poem",
        "news": "process_news",
        "joke": "process_joke",
    }
)

# Connect processors to text-to-speech
workflow.add_edge("process_general", "text_to_speech")
workflow.add_edge("process_poem", "text_to_speech")
workflow.add_edge("process_news", "text_to_speech")
workflow.add_edge("process_joke", "text_to_speech")

# Compile the graph
app = workflow.compile() 

## A function to convert text to a valid informative filename
def sanitize_filename(text, max_length=20):
    """Convert text to a valid and concise filename."""
    sanitized = re.sub(r'[^\w\s-]', '', text.lower())
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized[:max_length]

## Define Function to Run Agent and Play Audio

def run_tts_agent_and_play(input_text: str, content_type: str, save_file: bool = True):
    result = app.invoke({
        "input_text": input_text, 
        "processed_text": "", 
        "audio_data": b"",
        "audio_path": "", 
        "content_type": content_type
    })
    
    print(f"Detected content type: {result['content_type']}")
    print(f"Processed text: {result['processed_text']}")
    
    # Play the audio (this will only work in local Jupyter environment)
    display(Audio(result['audio_data'], autoplay=True))
    
    if save_file:
        # Create 'audio' directory in the parent folder of the notebook
        audio_dir = os.path.join('..', 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        sanitized_text = sanitize_filename(input_text)
        file_name = f"{content_type}_{sanitized_text}.mp3"
        file_path = os.path.join(audio_dir, file_name)
        
        with open(file_path, "wb") as f:
            f.write(result['audio_data'])
        
        print(f"Audio saved to: {file_path}")
        
        # Relative path for GitHub
        github_relative_path = f"../audio/{file_name}"
        display(Markdown(f"[Download {content_type} audio: {sanitized_text}]({github_relative_path})"))
        
        # Note about GitHub limitations
        print("Note: Audio playback is not supported directly on GitHub. Use the download link to listen to the audio.")
    else:
        print("Audio not saved to file.")
    
    return result

## Test the Text-to-Speech Agent

examples = {
    "general": "The quick brown fox jumps over the lazy dog.",
    "poem": "Biden encouraged Netanyahu to ‘design’ retaliation against Iran to deter attacks against Israel, senior US official says",
    "news": "Breaking news: Scientists discover a new species of deep-sea creature in the Mariana Trench.",
    "joke": "Why don't scientists trust atoms? Because they make up everything!"
}

for content_type, text in examples.items():
    print(f"\nProcessing example for {content_type} content:")
    print(f"Input text: {text}")
    
    # Run the TTS agent and save the file
    result = run_tts_agent_and_play(text, content_type, save_file=True)
    
    print("-" * 50)

print("All examples processed. You can download the audio files using the links above.")