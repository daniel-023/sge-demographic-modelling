# Utterance Table Guide

`scripts.preprocessing.build_utterance_table` builds utterance-level CSV files used by feature extraction. It supports the speaker-folder layout used for Parts 1 and 2, plus a Part 3 TextGrid mode for slicing long speaker-separate recordings into utterance WAVs.

Run commands from the repository root.

## Speaker-Folder Parts

Use this for parts where utterances already exist as individual WAV files under speaker folders.

```bash
python3 -m scripts.preprocessing.build_utterance_table \
    --parts 1 2 \
    --audio_root_base /path/to/audio/root
```

The script looks for layouts such as:

```text
part_1/speakers/<speaker>/wav/*.wav
part1/speakers/<speaker>/wav/*.wav
```

## Part 3 TextGrid Slicing

Use this when Part 3 has long speaker-separate WAV files and Praat/TextGrid interval timestamps.

```bash
python3 -m scripts.preprocessing.build_utterance_table \
    --parts 3 \
    --part3_use_textgrid \
    --part3_audio_dir /path/to/part3/conversation_wavs \
    --part3_textgrid_dir "/path/to/part3/Scripts Separate" \
    --part3_slice_out_dir /path/to/output/sliced_utterances
```

The TextGrid mode:

- reads `xmin`, `xmax`, and `text` interval fields
- skips empty intervals and silence tokens such as `<Z>` and `<S>`
- filters clips using `--min_duration_sec` and `--max_duration_sec`
- writes sliced WAVs under `<slice_out_dir>/<speaker_id>/wav/`
- records the source WAV, TextGrid path, transcript, and interval timestamps

Useful optional flags:

```bash
--part3_silence_tokens "<Z>" "<S>"
--part3_overwrite_slices
--min_duration_sec 3.0
--max_duration_sec 15.0
--no_attach_speaker_metadata
```

## Output

By default, per-part utterance tables are written to:

```text
data/metadata/utterances/part{N}_utterances.csv
```

Core columns:

- `part`
- `speaker_id`
- `utt_id`
- `path`
- `duration_sec`
- `split`
- `age`, `gender`, `ethnicity` when speaker metadata is attached

Part 3 TextGrid mode also includes:

- `start_sec`
- `end_sec`
- `source_path`
- `textgrid_path`
- `transcript`

Use `--combined` to also write:

```text
data/metadata/utterances/all_utterances.csv
```
