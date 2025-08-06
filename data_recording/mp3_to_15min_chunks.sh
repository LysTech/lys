#!/bin/bash

# Audio file splitter script
# Splits an mp3 file into 15-minute chunks as 16kHz mono WAV files in separate folders
# Usage: ./mp3_to_15min_chunks.sh <filename>
# Example: ./mp3_to_15min_chunks.sh my_podcast

# Check if filename argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No filename provided!"
    echo "Usage: $0 <filename>"
    echo "Example: $0 my_podcast"
    exit 1
fi

# Check if LYS_DATA_DIR is set
if [ -z "$LYS_DATA_DIR" ]; then
    echo "Error: LYS_DATA_DIR environment variable is not set!"
    echo "Please set it in your shell configuration (e.g., .bashrc, .zshrc):"
    echo "export LYS_DATA_DIR=\"/path/to/your/data/directory\""
    exit 1
fi

AUDIO_DIR="$LYS_DATA_DIR/assets/audio"
INPUT_BASENAME="$1"
INPUT_FILE="$AUDIO_DIR/$INPUT_BASENAME.mp3"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found!"
    echo "Please make sure your mp3 file is in: $AUDIO_DIR"
    exit 1
fi

# Extract the filename without extension for the subfolder name
FILENAME=$(basename "$INPUT_FILE")
FOLDER_BASE_NAME="${FILENAME%.*}"
OUTPUT_DIR="$AUDIO_DIR/$FOLDER_BASE_NAME"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not installed."
    echo "Install it with: sudo apt install ffmpeg (Ubuntu/Debian) or brew install ffmpeg (macOS)"
    exit 1
fi

# Get the total duration of the audio file
echo "Analyzing audio file..."
TOTAL_DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$INPUT_FILE")
TOTAL_SECONDS=$(echo "$TOTAL_DURATION" | cut -d'.' -f1)
CHUNK_SECONDS=900  # 15 mins = 900 secs

# Calculate number of chunks needed
NUM_CHUNKS=$(( (TOTAL_SECONDS + CHUNK_SECONDS - 1) / CHUNK_SECONDS ))

echo "Total duration: $(($TOTAL_SECONDS / 60)) minutes"
echo "Will create $NUM_CHUNKS chunks of 15 minutes each as 16kHz mono WAV files"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Split the audio file
for (( i=0; i<$NUM_CHUNKS; i++ )); do
    CHUNK_NUM=$((i + 1))
    FOLDER_NAME="$OUTPUT_DIR/chunk_$CHUNK_NUM"
    OUTPUT_FILE="$FOLDER_NAME/${FOLDER_BASE_NAME}_part_$CHUNK_NUM.wav"
    START_TIME=$((i * CHUNK_SECONDS))
    
    # Create folder if it doesn't exist
    mkdir -p "$FOLDER_NAME"
    
    echo "Creating $FOLDER_NAME..."
    
    # Use ffmpeg to extract the chunk and convert to 16kHz mono WAV
    ffmpeg -i "$INPUT_FILE" -ss $START_TIME -t $CHUNK_SECONDS \
           -ar 16000 -ac 1 -c:a pcm_s16le \
           -avoid_negative_ts make_zero "$OUTPUT_FILE" -y -loglevel quiet
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully created $OUTPUT_FILE (16kHz mono WAV)"
    else
        echo "✗ Failed to create $OUTPUT_FILE"
    fi
done

# Move the original mp3 file into the subfolder
echo ""
echo "Moving original mp3 file to subfolder..."
mv "$INPUT_FILE" "$OUTPUT_DIR/"
if [ $? -eq 0 ]; then
    echo "✓ Moved $INPUT_BASENAME.mp3 to $OUTPUT_DIR/"
else
    echo "✗ Failed to move original mp3 file"
fi

echo ""
echo "Audio splitting complete!"
echo "Created $NUM_CHUNKS folders with 16kHz mono WAV files: $(ls -d $OUTPUT_DIR/chunk_* | tr '\n' ' ')"
echo ""
echo "File details:"
for (( i=1; i<=NUM_CHUNKS; i++ )); do
    wav_file="$OUTPUT_DIR/chunk_$i/${FOLDER_BASE_NAME}_part_$i.wav"
    if [ -f "$wav_file" ]; then
        size=$(ls -lh "$wav_file" | awk '{print $5}')
        echo "  $wav_file ($size)"
    fi
done
