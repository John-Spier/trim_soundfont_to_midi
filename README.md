# `trim_soundfont_to_midi.py`

Trim an **SF2** or **DLS** so it only contains presets/instruments and samples/waves that are referenced by a specific **MIDI** file (based on Program Change + Bank Select).

## Requirements

- Python 3.8+
- Third-party: `mido` for MIDI parsing

Install dependency:

```bash
python -m pip install mido
```

## Usage

Trim SF2:

```bash
python scripts\trim_soundfont_to_midi.py "song.mid" "in.sf2" -o "out.sf2"
```

Trim DLS:

```bash
python scripts\trim_soundfont_to_midi.py "song.mid" "in.dls" -o "out.dls"
```

Dry run (don’t write output):

```bash
python scripts\trim_soundfont_to_midi.py "song.mid" "in.sf2" -o "out.sf2" --dry-run
```

Bank matching mode:

- `--bank-style mma` (default): compares 14-bit bank as \((MSB<<8)|LSB\)
- `--bank-style gm`: compares only MSB (7-bit)

```bash
python scripts\trim_soundfont_to_midi.py "song.mid" "in.sf2" -o "out.sf2" --bank-style gm
```

## Notes

- The script is intentionally “minimal RIFF”: it parses/writes just enough SF2/DLS structure to keep the referenced content.
- If your MIDI never sends Program Change (or uses unusual bank select patterns), the “used set” may be empty or incomplete.
