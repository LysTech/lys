#!/bin/bash

# Audio file splitter script
# Splits podcast.mp3 into 30-minute chunks as 16kHz mono WAV files in separate folders

INPUT_FILE="podcast.mp3"
CHUNK_DURATION="15:00"  # 30 minutes in MM:SS format

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found!"
    exit 1
fi

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
echo ""

# Split the audio file
for (( i=0; i<$NUM_CHUNKS; i++ )); do
    CHUNK_NUM=$((i + 1))
    FOLDER_NAME="chunk_$CHUNK_NUM"
    OUTPUT_FILE="$FOLDER_NAME/podcast_part_$CHUNK_NUM.wav"
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

echo ""
echo "Audio splitting complete!"
echo "Created $NUM_CHUNKS folders with 16kHz mono WAV files: $(ls -d chunk_* | tr '\n' ' ')"
echo ""
echo "File details:"
for (( i=1; i<=NUM_CHUNKS; i++ )); do
    wav_file="chunk_$i/podcast_part_$i.wav"
    if [ -f "$wav_file" ]; then
        size=$(ls -lh "$wav_file" | awk '{print $5}')
        echo "  $wav_file ($size)"
    fi
done
