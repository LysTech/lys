#!/bin/bash

# Whisper transcription script for chunked WAV files
# Transcribes 16kHz mono WAV files using whisper.cpp
# Usage: ./transcribe.sh <folder_name>
# Example: ./transcribe.sh dwarkesh_kotkin

# Check if folder name argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No folder name provided!"
    echo "Usage: $0 <folder_name>"
    echo "Example: $0 dwarkesh_kotkin"
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
FOLDER_NAME="$1"
CHUNK_BASE_DIR="$AUDIO_DIR/$FOLDER_NAME"

# Check if the folder exists
if [ ! -d "$CHUNK_BASE_DIR" ]; then
    echo "Error: Folder $CHUNK_BASE_DIR not found!"
    echo "Make sure you've run the audio splitter script first: ./mp3_to_15min_chunks.sh $FOLDER_NAME"
    exit 1
fi

# Dynamically construct paths based on LYS_DATA_DIR
# Assume whisper.cpp is in the same parent directory as the data directory
CODE_BASE_DIR=$(dirname "$(dirname "$LYS_DATA_DIR")")
WHISPER_DIR="$CODE_BASE_DIR/whisper.cpp"
MODEL_PATH="$WHISPER_DIR/models/ggml-large-v3-turbo.bin"
WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"

# Check if whisper directory exists
if [ ! -d "$WHISPER_DIR" ]; then
    echo "Error: Whisper directory not found at $WHISPER_DIR"
    echo "Please make sure whisper.cpp is installed in the same parent directory as your data directory"
    echo "Expected structure:"
    echo "  $(dirname "$CODE_BASE_DIR")/"
    echo "  ├── $(basename "$CODE_BASE_DIR")/"
    echo "  │   ├── whisper.cpp/"
    echo "  │   └── $(basename "$LYS_DATA_DIR")/"
    exit 1
fi

# Check if whisper binary exists
if [ ! -f "$WHISPER_BIN" ]; then
    echo "Error: Whisper binary not found at $WHISPER_BIN"
    echo "Please make sure whisper.cpp is built:"
    echo "  cd $WHISPER_DIR"
    echo "  make"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please download the model:"
    echo "  cd $WHISPER_DIR"
    echo "  bash ./models/download-ggml-model.sh large-v3-turbo"
    exit 1
fi

# Find all chunk directories
CHUNK_DIRS=($(ls -d "$CHUNK_BASE_DIR"/chunk_* 2>/dev/null | sort -V))

if [ ${#CHUNK_DIRS[@]} -eq 0 ]; then
    echo "Error: No chunk directories found in $CHUNK_BASE_DIR!"
    echo "Make sure you've run the audio splitter script first: ./mp3_to_15min_chunks.sh $FOLDER_NAME"
    exit 1
fi

echo "Found ${#CHUNK_DIRS[@]} chunk directories in $CHUNK_BASE_DIR"
echo "Model: $MODEL_PATH"
echo "Whisper binary: $WHISPER_BIN"
echo ""

# Process each chunk directory
for chunk_dir in "${CHUNK_DIRS[@]}"; do
    echo "Processing $chunk_dir..."
    
    # Find WAV files in the chunk directory
    wav_files=($(find "$chunk_dir" -name "*.wav"))
    
    if [ ${#wav_files[@]} -eq 0 ]; then
        echo "  ⚠ No WAV files found in $chunk_dir"
        continue
    fi
    
    # Process each WAV file in the directory
    for wav_file in "${wav_files[@]}"; do
        filename=$(basename "$wav_file")
        
        echo "  Transcribing: $filename"
        
        # Run whisper transcription on the WAV file
        "$WHISPER_BIN" -m "$MODEL_PATH" -f "$wav_file" -oj -ml 1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully transcribed $filename"
            
            # Check if JSON output was created
            json_file="${wav_file%.*}.json"
            if [ -f "$json_file" ]; then
                echo "  ✓ Transcription saved to $(basename "$json_file")"
            fi
            
        else
            echo "  ✗ Failed to transcribe $filename"
        fi
        
        echo ""
    done
done

echo "Transcription complete!"
echo ""
echo "Summary of generated files:"
find "$CHUNK_BASE_DIR" -name "*.json" 2>/dev/null | while read json_file; do
    echo "  $json_file"
done

# Count total transcriptions
total_json=$(find "$CHUNK_BASE_DIR" -name "*.json" 2>/dev/null | wc -l)
echo ""
echo "Total transcriptions created: $total_json"

# Show what's in each chunk directory
echo ""
echo "Contents of chunk directories:"
for chunk_dir in "${CHUNK_DIRS[@]}"; do
    echo "$(basename "$chunk_dir"):"
    ls -lh "$chunk_dir/" | grep -E '\.(wav|json)$' | awk '{print "  " $9 " (" $5 ")"}'
done
