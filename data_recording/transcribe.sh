#!/bin/bash

# Whisper transcription script for chunked WAV files
# Transcribes 16kHz mono WAV files using whisper.cpp

MODEL_PATH="/Users/thomasrialan/Documents/code/whisper.cpp/models/ggml-large-v3-turbo.bin"
WHISPER_BIN="/Users/thomasrialan/Documents/code/whisper.cpp/build/bin/whisper-cli"

# Check if whisper binary exists
if [ ! -f "$WHISPER_BIN" ]; then
    echo "Error: Whisper binary not found at $WHISPER_BIN"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Find all chunk directories
CHUNK_DIRS=($(ls -d chunk_* 2>/dev/null | sort -V))

if [ ${#CHUNK_DIRS[@]} -eq 0 ]; then
    echo "Error: No chunk directories found!"
    echo "Make sure you've run the audio splitter script first."
    exit 1
fi

echo "Found ${#CHUNK_DIRS[@]} chunk directories"
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
find chunk_* -name "*.json" 2>/dev/null | while read json_file; do
    echo "  $json_file"
done

# Count total transcriptions
total_json=$(find chunk_* -name "*.json" 2>/dev/null | wc -l)
echo ""
echo "Total transcriptions created: $total_json"

# Show what's in each chunk directory
echo ""
echo "Contents of chunk directories:"
for chunk_dir in "${CHUNK_DIRS[@]}"; do
    echo "$chunk_dir:"
    ls -lh "$chunk_dir/" | grep -E '\.(wav|json)$' | awk '{print "  " $9 " (" $5 ")"}'
done
