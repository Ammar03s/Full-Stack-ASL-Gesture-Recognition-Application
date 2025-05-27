# ASL

This application allows you to compose sentences using American Sign Language (ASL). The system detects hand signs through your webcam and automatically adds detected letters to your sentence every 2 seconds.

## Features

- Real-time ASL sign language detection
- Builds sentences from detected signs
- 2-second interval between letter detections
- Ability to pause detection (Freeze)
- Manual controls for spaces and backspace
- Save sentences to a JSON file

## How to Use

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python asl_sentence_builder.py
   ```

3. Click "Start Detection" to begin

4. Controls:
   - Use the "Add Space" button to manually add spaces
   - Use "Backspace" to delete the last character
   - Use "Clear Sentence" to start over
   - Use "Freeze" to pause detection temporarily
   - Use "Save Sentence" to save your completed text to a file

## Special Signs

- "space": Adds a space to your sentence
- "del": Deletes the last character (same as backspace)
- "nothing": No action (used when transitioning between signs)

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in requirements.txt

## Saving Sentences

Sentences are saved to a JSON file named "current_sentence.json" in the "saved_sentences" directory. Each time you save a new sentence, it will replace the previous one. 