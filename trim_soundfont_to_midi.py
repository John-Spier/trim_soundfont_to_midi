#!/usr/bin/env python3
"""
Trim an SF2/SF3 or DLS file down to what a specific MIDI file actually plays.

For every note-on in the MIDI, the bank/program active on that part picks a
preset. Only presets that sound a note are kept, and inside them only the preset
zones / instrument zones / DLS regions whose key and velocity ranges are hit by a
played note, plus the samples/waves those reference. When different players
would interpret the MIDI differently, everything any of them could play is kept.

Usage:
  python trim_soundfont_to_midi.py song.mid soundfont.sf2 -o out.sf2
  python trim_soundfont_to_midi.py song.mid collection.dls -o out.dls
  python trim_soundfont_to_midi.py song.mid soundfont.sf2 -o out.sf2 --dry-run

MIDI support (built-in parser, no dependencies):
  SMF type 0/1/2, RMID (RIFF-wrapped) files, junk before/between chunks, wrong track
  lengths, truncated tracks, running status, multi-port files (FF 21 port meta).
  GM / GM2 System On, GS Reset / System Mode Set, XG System On / All Parameter Reset.
  GS: Use-for-Rhythm-Part, Tone Number (bank+program) and Rx Channel per part, part and
      master Key Shift, port B parts (50 xx xx).
  XG: Part Mode (drum), Bank MSB/LSB / Program / Rx Channel per part, Note Shift,
      Master Transpose, MSB 120-127 drum voices on any channel.
  GM2: MSB 120 = rhythm, MSB 121 = melodic (variation in LSB), Master Coarse Tuning.
  Notes sounding on the same tick as a program change are counted for both programs.
  Transposed notes are kept alongside the untransposed ones (players differ).

Bank matching (--bank-style), SF2:
  auto (default) Follow the file: GS/no sysex -> bank = CC0 (MSB); XG / GM2 -> bank = CC32
                 (LSB) when MSB is 0/121, else MSB; after GM System On, bank 0 too.
  msb  (alias gs, gm) bank = CC0.     lsb  bank = CC32.     mma  bank = CC0*128 + CC32.
  xg   force XG rules from the start. none ignore bank select (always bank 0).
  all  union of every interpretation above.
DLS addresses instruments by MSB + LSB + drum flag natively, so it always matches both
(none -> bank 0; all/GM -> bank 0 as well).

Drums: a drum part maps to SF2 bank 128 / DLS drum instruments when the soundfont has
any; otherwise it is treated like a melodic part.
Missing presets fall back like FluidSynth: bank 0 same program, then bank 0 program 0;
drum kits fall back to kit 0.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Iterable, List, NamedTuple, Optional, Set, Tuple

DRUM_CHANNEL = 9  # MIDI channel 10 (0-based)
DRUM_BANK = 128   # SF2 percussion bank

Note = Tuple[int, int]  # (key, velocity)


class Sel(NamedTuple):
    """A program selection as the MIDI made it, before mapping to a soundfont."""
    msb: int
    lsb: int
    prog: int
    drum: bool
    mode: str  # 'gs' (also no sysex), 'gm', 'gm2', 'xg'


Usage = Dict[Sel, Set[Note]]


# ---------------------------------------------------------------------------
# RIFF helpers
# ---------------------------------------------------------------------------

class RiffError(Exception):
    pass


@dataclass
class Chunk:
    fourcc: str
    data: memoryview
    children: Optional[List["Chunk"]] = None
    list_type: Optional[str] = None


def _read_u32le(buf, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _read_u16le(buf, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def _parse_riff(buf: bytes) -> Chunk:
    mv = memoryview(buf)
    if len(mv) < 12 or mv[0:4].tobytes() != b"RIFF":
        raise RiffError("Not a RIFF file")
    size = _read_u32le(mv, 4)
    root_data = mv[12: min(len(mv), 8 + size)]  # tolerate a wrong RIFF size
    root = Chunk("RIFF", root_data, list_type=mv[8:12].tobytes().decode("ascii", "replace"))
    root.children = _parse_children(root_data, 0, len(root_data))
    return root


def _parse_children(mv: memoryview, start: int, end: int) -> List[Chunk]:
    out: List[Chunk] = []
    off = start
    while off + 8 <= end:
        fourcc = mv[off: off + 4].tobytes().decode("ascii", "replace")
        size = _read_u32le(mv, off + 4)
        data_start = off + 8
        data_end = data_start + size
        if data_end > end:
            break
        ch = Chunk(fourcc, mv[data_start:data_end])
        if fourcc in ("RIFF", "LIST") and size >= 4:
            ch.list_type = ch.data[0:4].tobytes().decode("ascii", "replace")
            ch.children = _parse_children(ch.data, 4, len(ch.data))
        out.append(ch)
        off = data_end + (size & 1)
    return out


def _find_list(root: Chunk, list_type: str) -> Optional[Chunk]:
    if root.fourcc in ("RIFF", "LIST") and root.list_type == list_type:
        return root
    for c in root.children or []:
        got = _find_list(c, list_type)
        if got:
            return got
    return None


def _find_child(parent: Chunk, fourcc: str) -> Optional[Chunk]:
    for c in parent.children or []:
        if c.fourcc == fourcc:
            return c
    return None


def _is_list(c: Chunk, list_type: str) -> bool:
    return c.fourcc == "LIST" and c.list_type == list_type


def _read_zstr(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("ascii", "replace")


def _riff_chunk(fourcc: str, data: bytes) -> bytes:
    """fourcc + size + data, padded to an even length as RIFF requires."""
    return fourcc.encode("ascii")[:4] + struct.pack("<I", len(data)) + data + (b"\x00" * (len(data) & 1))


def _riff_list(list_type: str, body: bytes) -> bytes:
    return _riff_chunk("LIST", list_type.encode("ascii")[:4] + body)


def _raw(c: Chunk) -> bytes:
    """Re-serialize an unchanged chunk (LIST data already starts with its list type)."""
    return _riff_chunk(c.fourcc, c.data.tobytes())


# ---------------------------------------------------------------------------
# MIDI file reading (tolerant)
# ---------------------------------------------------------------------------

class MidiError(Exception):
    pass


Event = Tuple[int, int, bytes]  # (absolute tick, status byte, data)


def _varlen(d: bytes, p: int, end: int) -> Tuple[int, int]:
    v = 0
    for _ in range(4):
        if p >= end:
            raise IndexError
        b = d[p]
        p += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            break
    return v, p


def _parse_track(d: bytes, p: int, end: int) -> Tuple[List[Event], int, bool]:
    """Returns (events, position after End-of-Track or where parsing stopped, saw EOT)."""
    ev: List[Event] = []
    t = 0
    running = 0
    try:
        while p < end:
            delta, p = _varlen(d, p, end)
            t += delta
            if p >= end:
                break
            b = d[p]
            if b == 0xFF:  # meta (running status deliberately kept: many files rely on it)
                if p + 1 >= end:
                    break
                typ = d[p + 1]
                ln, q = _varlen(d, p + 2, end)
                ev.append((t, 0xFF, bytes([typ]) + d[q:q + ln]))
                p = q + ln
                if typ == 0x2F:
                    return ev, p, True
                continue
            if b in (0xF0, 0xF7):
                ln, q = _varlen(d, p + 1, end)
                ev.append((t, b, d[q:q + ln]))
                p = q + ln
                continue
            if b & 0x80:
                p += 1
                if b >= 0xF0:  # stray system common / realtime byte inside a file
                    p += {0xF1: 1, 0xF2: 2, 0xF3: 1}.get(b, 0)
                    continue
                running = status = b
            else:
                if not running:
                    p += 1  # data byte with no status: skip garbage
                    continue
                status = running
            n = 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
            args = bytes(min(x, 0x7F) for x in d[p:p + n])  # some rips store velocity 128: clip, don't mask
            p += n
            if len(args) < n:
                break
            ev.append((t, status, args))
    except IndexError:
        pass
    return ev, p, False


def read_midi(path: str) -> Tuple[int, List[List[Event]]]:
    """Return (SMF format, tracks). Accepts RMID and damaged files."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == b"RIFF" and data[8:12] == b"RMID":
        try:
            c = _find_child(_parse_riff(data), "data")
            if c is not None:
                data = c.data.tobytes()
        except RiffError:
            pass
    start = data.find(b"MThd")
    if start < 0 or start + 14 > len(data):
        raise MidiError("no MIDI header (MThd) found")
    hlen = struct.unpack_from(">I", data, start + 4)[0]
    fmt = struct.unpack_from(">H", data, start + 8)[0]
    pos = start + 8 + max(6, hlen if hlen < 0x10000 else 6)
    tracks: List[List[Event]] = []
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        ln = struct.unpack_from(">I", data, pos + 4)[0]
        if cid != b"MTrk":
            if all(0x20 <= b < 0x7F for b in cid) and pos + 8 + ln <= len(data):
                pos += 8 + ln  # unknown but well-formed chunk
                continue
            nxt = data.find(b"MTrk", pos + 1)
            if nxt < 0:
                break
            pos = nxt
            continue
        declared_end = pos + 8 + ln
        events, stop, eot = _parse_track(data, pos + 8, min(declared_end, len(data)))
        if not eot and declared_end < len(data) and data[declared_end:declared_end + 4] != b"MTrk":
            # Declared length too short: keep reading up to the next MTrk / end of file.
            nxt = data.find(b"MTrk", pos + 8)
            events, stop, eot = _parse_track(data, pos + 8, nxt if nxt > 0 else len(data))
        tracks.append(events)
        if eot and stop != declared_end and data[stop:stop + 4] == b"MTrk":
            pos = stop  # declared length wrong, but End-of-Track is followed by the next track
        else:
            pos = max(declared_end, pos + 8)
    if not tracks:
        raise MidiError("no MIDI tracks (MTrk) found")
    return fmt, tracks


# ---------------------------------------------------------------------------
# MIDI interpretation: which (bank, program, drum) plays which notes
# ---------------------------------------------------------------------------

def _gs_block_to_part(block: int) -> int:
    """GS part address block 1x: 0 = part 10, 1-9 = parts 1-9, A-F = parts 11-16."""
    if block == 0:
        return 9
    if block <= 9:
        return block - 1
    return block


@dataclass
class _Part:
    port: int
    idx: int
    rx: Optional[Tuple[int, int]]  # (port, channel) it listens to, None = off
    msb: int = 0
    lsb: int = 0
    prog: int = 0
    rhythm: Optional[bool] = None  # explicit drum/normal set by GS/XG sysex
    key_shift: int = 0
    sels: List[Sel] = field(default_factory=list)


class _MidiInterpreter:
    def __init__(self, force_mode: Optional[str] = None):
        self.force_mode = force_mode
        self.parts: Dict[Tuple[int, int], _Part] = {}
        self.mode: Dict[int, str] = {}
        self.master_shift: Dict[int, int] = {}
        self.usage: Usage = {}
        self._rx_cache: Optional[Dict[Tuple[int, int], List[_Part]]] = None

    # -- state --
    def _mode(self, port: int) -> str:
        return self.force_mode or self.mode.get(port, "gs")

    def _ensure_port(self, port: int) -> None:
        if (port, 0) not in self.parts:
            self._reset_parts(port)

    def _reset_parts(self, port: int) -> None:
        for i in range(16):
            p = _Part(port=port, idx=i, rx=(port, i))
            self.parts[(port, i)] = p
            self._select(p)
        self._rx_cache = None

    def _reset(self, port: int, mode: str) -> None:
        if not self.force_mode:
            self.mode[port] = mode
        self.master_shift[port] = 0
        self._reset_parts(port)

    def _part(self, port: int, idx: int) -> _Part:
        self._ensure_port(port)
        return self.parts[(port, idx)]

    def _listeners(self, port: int, ch: int) -> List[_Part]:
        self._ensure_port(port)
        if self._rx_cache is None:
            cache: Dict[Tuple[int, int], List[_Part]] = {}
            for p in self.parts.values():
                if p.rx is not None:
                    cache.setdefault(p.rx, []).append(p)
            self._rx_cache = cache
        return self._rx_cache.get((port, ch), [])

    def _select(self, p: _Part) -> None:
        """Apply the part's bank/program (Program Change, or sysex tone/part-mode change)."""
        mode = self._mode(p.port)
        default_drum = p.idx == DRUM_CHANNEL if p.rhythm is None else p.rhythm
        drums: List[bool]
        if mode == "xg":
            if p.msb >= 120:  # XG 126/127 drum kits (FluidSynth uses >= 120)
                drums = [True]
            elif p.rhythm is not None:
                drums = [p.rhythm]
            elif default_drum:
                drums = [True, False]  # XG ch10 + MSB<120 switches to a normal voice; not all players do
            else:
                drums = [False]
        elif mode == "gm2":
            drums = [True] if p.msb == 120 else [False] if p.msb == 121 else [default_drum]
        else:  # GS / GM: drum parts ignore bank select
            drums = [default_drum]
        p.sels = [Sel(p.msb, p.lsb, p.prog, d, mode) for d in drums]

    # -- events --
    def _note(self, p: _Part, key: int, vel: int) -> None:
        shift = self.master_shift.get(p.port, 0) + p.key_shift
        for s in p.sels:
            notes = self.usage.setdefault(s, set())
            notes.add((key, vel))
            if shift and not s.drum and 0 <= key + shift <= 127:
                notes.add((key + shift, vel))

    def _sysex(self, port: int, d: bytes) -> None:
        if d[:1] == b"\xF0":
            d = d[1:]
        if d[-1:] == b"\xF7":
            d = d[:-1]
        if len(d) < 4:
            return
        man = d[0]
        if man == 0x7E and d[2] == 0x09:  # GM System On / Off, GM2 System On
            mode = {1: "gm", 2: "gs", 3: "gm2"}.get(d[3])
            if mode:
                self._reset(port, mode)
        elif man == 0x7F and len(d) >= 6 and d[2] == 0x04 and d[3] == 0x04:  # Master Coarse Tuning
            self.master_shift[port] = d[5] - 0x40
        elif man == 0x41 and len(d) >= 8 and d[2] == 0x42 and d[3] == 0x12:  # Roland GS DT1
            base = (d[4] << 14) | (d[5] << 7) | d[6]
            for i, v in enumerate(d[7:-1]):  # last byte is the checksum
                self._gs(port, base + i, v)
        elif man == 0x43 and len(d) >= 7 and (d[1] & 0xF0) == 0x10 and d[2] == 0x4C:  # Yamaha XG
            for i, v in enumerate(d[6:]):
                self._xg(port, d[3], d[4], d[5] + i, v)

    def _gs(self, port: int, addr: int, v: int) -> None:
        a0, a1, a2 = (addr >> 14) & 0x7F, (addr >> 7) & 0x7F, addr & 0x7F
        if (a0, a1, a2) in ((0x40, 0x00, 0x7F), (0x00, 0x00, 0x7F)):  # GS Reset / System Mode Set
            self._reset(port, "gs")
        elif (a0, a1, a2) == (0x40, 0x00, 0x05):  # Master Key Shift
            self.master_shift[port] = v - 0x40
        elif a0 in (0x40, 0x50) and (a1 & 0xF0) == 0x10:  # part parameters (0x50 = port B)
            pport = port + (a0 == 0x50)
            p = self._part(pport, _gs_block_to_part(a1 & 0x0F))
            if a2 == 0x00:    # Tone Number: CC#0 value
                p.msb = v
            elif a2 == 0x01:  # Tone Number: program
                p.prog = v
                self._select(p)
            elif a2 == 0x02:  # Rx Channel (0x10 = off)
                p.rx = (pport, v) if v < 16 else None
                self._rx_cache = None
            elif a2 == 0x15:  # Use for Rhythm Part (0 = off, 1/2 = drum map)
                p.rhythm = v != 0
                self._select(p)
            elif a2 == 0x16:  # Pitch Key Shift
                p.key_shift = v - 0x40

    def _xg(self, port: int, ah: int, am: int, al: int, v: int) -> None:
        if ah == 0x00 and am == 0x00:
            if al in (0x7E, 0x7F):  # XG System On / All Parameter Reset
                self._reset(port, "xg")
            elif al == 0x06:  # Master Transpose
                self.master_shift[port] = v - 0x40
        elif ah == 0x08:  # Multi Part
            pport = port + am // 16
            p = self._part(pport, am % 16)
            if al == 0x01:
                p.msb = v
            elif al == 0x02:
                p.lsb = v
            elif al == 0x03:
                p.prog = v
                self._select(p)
            elif al == 0x04:  # Rx Channel (0x7F = off)
                p.rx = None if v == 0x7F else (port + v // 16, v % 16)
                self._rx_cache = None
            elif al == 0x07:  # Part Mode (0 = normal, else drum)
                p.rhythm = v != 0
                self._select(p)
            elif al == 0x08:  # Note Shift
                p.key_shift = v - 0x40

    def run(self, fmt: int, tracks: List[List[Event]]) -> Usage:
        merged = []
        offset = 0
        for ti, tr in enumerate(tracks):
            merged.extend((t + offset, ti, seq, st, data) for seq, (t, st, data) in enumerate(tr))
            if fmt == 2 and tr:
                offset += tr[-1][0]  # type 2: patterns play one after another
        merged.sort(key=lambda e: (e[0], e[1], e[2]))

        track_port = [0] * len(tracks)
        i = 0
        while i < len(merged):
            tick = merged[i][0]
            played: List[Tuple[_Part, List[Sel], int, int]] = []
            while i < len(merged) and merged[i][0] == tick:
                _, ti, _, st, data = merged[i]
                i += 1
                port = track_port[ti]
                if st == 0xFF:
                    if data[:1] == b"\x21" and len(data) >= 2:
                        track_port[ti] = data[1] & 0x7F
                    continue
                if st in (0xF0, 0xF7):
                    self._sysex(port, data)
                    continue
                kind, ch = st & 0xF0, st & 0x0F
                if kind == 0xB0:
                    for p in self._listeners(port, ch):
                        if data[0] == 0:
                            p.msb = data[1]
                        elif data[0] == 32:
                            p.lsb = data[1]
                elif kind == 0xC0:
                    for p in self._listeners(port, ch):
                        p.prog = data[0]
                        self._select(p)
                elif kind == 0x90 and data[1] > 0:
                    for p in self._listeners(port, ch):
                        self._note(p, data[0], data[1])
                        played.append((p, p.sels, data[0], data[1]))
            # A program change on the same tick as a note is ordered differently by different
            # players; count the note for the program in effect at the end of the tick as well.
            for p, sels, key, vel in played:
                if p.sels is not sels:
                    self._note(p, key, vel)
        return self.usage


def collect_midi_usage(midi_path: str, bank_style: str = "auto") -> Usage:
    """Return {Sel: {(key, velocity), ...}} for every sounding note."""
    fmt, tracks = read_midi(midi_path)
    usage = _MidiInterpreter("xg" if bank_style == "xg" else None).run(fmt, tracks)
    if bank_style == "all":
        for sel, notes in _MidiInterpreter("xg").run(fmt, tracks).items():
            usage.setdefault(sel, set()).update(notes)
    return usage


# ---------------------------------------------------------------------------
# Mapping selections onto soundfont programs
# ---------------------------------------------------------------------------

def sf2_banks(sel: Sel, style: str) -> List[int]:
    m, l = sel.msb, sel.lsb
    variation = l if m in (0, 121) else m  # XG / GM2: variation lives in the LSB
    if style == "auto":
        if sel.mode in ("xg", "gm2"):
            return [variation]
        if sel.mode == "gm":
            return [m, 0] if m else [0]  # GM devices ignore bank select
        return [m]
    if style == "lsb":
        return [l]
    if style == "mma":
        return [(m << 7) | l]
    if style == "xg":
        return [variation]
    if style == "none":
        return [0]
    if style == "all":
        return list(dict.fromkeys([m, l, (m << 7) | l, variation, 0]))
    return [m]  # msb / gs / gm


def _sel_label(sel: Sel) -> str:
    return f"{'drum ' if sel.drum else ''}bank {sel.msb}/{sel.lsb} program {sel.prog} ({sel.mode.upper()})"


def resolve_usage(
    usage: Usage,
    available: Set[Hashable],
    candidates: Callable[[Sel], List[List[Hashable]]],
    describe: Callable[[Hashable], str],
) -> Tuple[Dict[Hashable, Set[Note]], List[str]]:
    """Each selection yields one or more fallback chains; the first available entry of each is used."""
    out: Dict[Hashable, Set[Note]] = {}
    msgs: Dict[str, None] = {}
    for sel, notes in sorted(usage.items()):
        for chain in candidates(sel):
            got = next((c for c in chain if c in available), None)
            if got is None:
                msgs[f"Warning: nothing in the soundfont for {_sel_label(sel)}; those notes are silent."] = None
                continue
            if got != chain[0]:
                msgs[f"Note: {_sel_label(sel)} is missing; players fall back to {describe(got)}."] = None
            out.setdefault(got, set()).update(notes)
    return out, list(msgs)


# ---------------------------------------------------------------------------
# SF2 / SF3
# ---------------------------------------------------------------------------

GEN_KEY_RANGE = 43
GEN_VEL_RANGE = 44
GEN_INSTRUMENT = 41
GEN_SAMPLE_ID = 53

SAMPLE_PAD = 46            # zero sample points after each sample (SF2 spec minimum)
SAMPLE_LINK_TYPES = 0x0E   # right / left / linked
SAMPLE_COMPRESSED = 0x10   # SF3 Ogg Vorbis
SAMPLE_ROM = 0x8000


@dataclass
class SF2Zone:
    gens: List[Tuple[int, int]]  # (operator, amount as raw u16)
    mods: List[bytes]            # raw 10-byte modulator records

    def get(self, oper: int) -> Optional[int]:
        for o, a in self.gens:
            if o == oper:
                return a
        return None


@dataclass
class SF2Header:
    raw: bytes        # full record (phdr 38 / inst 22 bytes)
    zones: List[SF2Zone] = field(default_factory=list)


@dataclass
class SF2Sample:
    raw: bytes        # 46-byte shdr record
    start: int
    end: int
    start_loop: int
    end_loop: int
    link: int
    stype: int


@dataclass
class SF2Parsed:
    presets: List[SF2Header]     # without terminal EOP
    insts: List[SF2Header]       # without terminal EOI
    samples: List[SF2Sample]     # without terminal EOS
    smpl: bytes
    sm24: Optional[bytes]
    info: List[Tuple[str, bytes]]


def _parse_sf2(path: str) -> SF2Parsed:
    with open(path, "rb") as f:
        root = _parse_riff(f.read())
    if root.list_type != "sfbk":
        raise RiffError("Not an SF2 (RIFF sfbk)")
    sdta = _find_list(root, "sdta")
    pdta = _find_list(root, "pdta")
    if not sdta or not pdta:
        raise RiffError("Malformed SF2: missing sdta/pdta")
    smpl = _find_child(sdta, "smpl")
    sm24 = _find_child(sdta, "sm24")

    def get(name: str, required: bool = True) -> memoryview:
        c = _find_child(pdta, name)
        if not c:
            if required:
                raise RiffError(f"Malformed SF2: missing pdta/{name}")
            return memoryview(b"")
        return c.data

    def records(d: memoryview, size: int) -> List[bytes]:
        return [d[o: o + size].tobytes() for o in range(0, len(d) - size + 1, size)]

    phdr = records(get("phdr"), 38)
    pbag = [struct.unpack("<HH", r) for r in records(get("pbag"), 4)]
    pmod = records(get("pmod", False), 10)
    pgen = [struct.unpack("<HH", r) for r in records(get("pgen"), 4)]
    inst = records(get("inst"), 22)
    ibag = [struct.unpack("<HH", r) for r in records(get("ibag"), 4)]
    imod = records(get("imod", False), 10)
    igen = [struct.unpack("<HH", r) for r in records(get("igen"), 4)]
    shdr = records(get("shdr"), 46)
    if len(phdr) < 2 or len(inst) < 2:
        raise RiffError("Malformed SF2: no presets or instruments")

    def build(hdrs: List[bytes], bag_off: int, bags, gens, mods) -> List[SF2Header]:
        out = []
        for i in range(len(hdrs) - 1):  # last is the terminal record
            b0 = _read_u16le(hdrs[i], bag_off)
            b1 = _read_u16le(hdrs[i + 1], bag_off)
            h = SF2Header(raw=hdrs[i])
            for b in range(b0, min(b1, len(bags) - 1)):
                g0, m0 = bags[b]
                g1, m1 = bags[b + 1]
                h.zones.append(SF2Zone(gens=list(gens[g0:g1]), mods=list(mods[m0:m1])))
            out.append(h)
        return out

    samples = []
    for r in shdr[:-1] if shdr else []:
        start, end, sl, el = struct.unpack_from("<IIII", r, 20)
        link, stype = struct.unpack_from("<HH", r, 42)
        samples.append(SF2Sample(r, start, end, sl, el, link, stype))

    info: List[Tuple[str, bytes]] = []
    info_list = _find_list(root, "INFO")
    for c in (info_list.children if info_list else None) or []:
        if c.fourcc not in ("RIFF", "LIST"):
            info.append((c.fourcc, c.data.tobytes()))

    return SF2Parsed(
        presets=build(phdr, 24, pbag, pgen, pmod),
        insts=build(inst, 20, ibag, igen, imod),
        samples=samples,
        smpl=smpl.data.tobytes() if smpl else b"",
        sm24=sm24.data.tobytes() if sm24 else None,
        info=info,
    )


def _split_global(zones: List[SF2Zone], terminal_gen: int) -> Tuple[Optional[SF2Zone], List[SF2Zone]]:
    """Per spec, a first zone without the terminal generator is the global zone; later ones are ignored."""
    glob = zones[0] if zones and zones[0].get(terminal_gen) is None else None
    local = [z for z in (zones[1:] if glob else zones) if z.get(terminal_gen) is not None]
    return glob, local


def _in_range(amount: Optional[int], value: int) -> bool:
    if amount is None:
        return True
    return (amount & 0xFF) <= value <= ((amount >> 8) & 0xFF)


def _zone_hits(zone: SF2Zone, glob: Optional[SF2Zone], notes: Iterable[Note]) -> Set[Note]:
    kr = zone.get(GEN_KEY_RANGE)
    vr = zone.get(GEN_VEL_RANGE)
    if glob is not None:  # global zone ranges are defaults for local zones
        kr = glob.get(GEN_KEY_RANGE) if kr is None else kr
        vr = glob.get(GEN_VEL_RANGE) if vr is None else vr
    return {(n, v) for n, v in notes if _in_range(kr, n) and _in_range(vr, v)}


@dataclass
class SF2Trimmed:
    presets: List[Tuple[SF2Header, List[SF2Zone]]]  # header + zones to keep (global first if any)
    insts: List[Tuple[SF2Header, List[SF2Zone]]]
    inst_map: Dict[int, int]
    sample_list: List[int]
    sample_map: Dict[int, int]


def _sf2_trim(sf2: SF2Parsed, usage: Usage, style: str) -> Tuple[SF2Trimmed, List[str]]:
    n_inst = len(sf2.insts)
    n_smp = len(sf2.samples)

    lookup: Dict[Tuple[int, int], List[int]] = {}
    for i, p in enumerate(sf2.presets):
        prog, bank = struct.unpack_from("<HH", p.raw, 20)
        lookup.setdefault((bank, prog), []).append(i)  # duplicates: players differ on which wins, keep all
    has_drum_bank = any(b == DRUM_BANK for b, _ in lookup)

    def candidates(sel: Sel) -> List[List[Hashable]]:
        if sel.drum and has_drum_bank:
            return [[(DRUM_BANK, sel.prog), (DRUM_BANK, 0)]]
        return [[(b, sel.prog), (0, sel.prog), (0, 0)] for b in sf2_banks(sel, style)]

    prog_notes, msgs = resolve_usage(usage, set(lookup), candidates, lambda k: f"bank {k[0]} program {k[1]}")

    # Presets -> preset zones hit by notes -> notes that reach each instrument.
    preset_keep: Dict[int, Tuple[Optional[SF2Zone], List[SF2Zone]]] = {}
    inst_notes: Dict[int, Set[Note]] = {}
    for key, notes in prog_notes.items():
        for pi in lookup[key]:  # type: ignore[index]
            glob, local = _split_global(sf2.presets[pi].zones, GEN_INSTRUMENT)
            kept = []
            for z in local:
                ii = z.get(GEN_INSTRUMENT)
                if ii is None or ii >= n_inst:
                    continue
                hits = _zone_hits(z, glob, notes)
                if hits:
                    kept.append(z)
                    inst_notes.setdefault(ii, set()).update(hits)
            preset_keep[pi] = (glob, kept)

    # Instruments -> instrument zones hit by notes -> samples.
    inst_keep: Dict[int, Tuple[Optional[SF2Zone], List[SF2Zone]]] = {}
    used_samples: Set[int] = set()
    for ii, notes in inst_notes.items():
        glob, local = _split_global(sf2.insts[ii].zones, GEN_SAMPLE_ID)
        kept = []
        for z in local:
            si = z.get(GEN_SAMPLE_ID)
            if si is None or si >= n_smp:
                continue
            if _zone_hits(z, glob, notes):
                kept.append(z)
                used_samples.add(si)
        if kept:
            inst_keep[ii] = (glob, kept)

    # Stereo / linked samples need their partner.
    stack = list(used_samples)
    while stack:
        s = sf2.samples[stack.pop()]
        if s.stype & SAMPLE_LINK_TYPES and s.link < n_smp and s.link not in used_samples:
            used_samples.add(s.link)
            stack.append(s.link)

    inst_list = sorted(inst_keep)
    inst_map = {old: new for new, old in enumerate(inst_list)}
    sample_list = sorted(used_samples)
    sample_map = {old: new for new, old in enumerate(sample_list)}

    presets = []
    for pi in sorted(preset_keep):
        glob, kept = preset_keep[pi]
        kept = [z for z in kept if z.get(GEN_INSTRUMENT) in inst_map]
        if kept:
            presets.append((sf2.presets[pi], ([glob] if glob else []) + kept))
    if not presets:
        raise ValueError("\n".join(["No preset in the soundfont is played by the MIDI file."] + msgs))

    insts = [(sf2.insts[ii], ([inst_keep[ii][0]] if inst_keep[ii][0] else []) + inst_keep[ii][1]) for ii in inst_list]
    return SF2Trimmed(presets, insts, inst_map, sample_list, sample_map), msgs


def _ordered_gens(gens: List[Tuple[int, int]], terminal: int, remap: Dict[int, int]) -> List[Tuple[int, int]]:
    """Spec order: keyRange first, velRange second, instrument/sampleID last (some players drop others)."""
    kr = [g for g in gens if g[0] == GEN_KEY_RANGE][:1]
    vr = [g for g in gens if g[0] == GEN_VEL_RANGE][:1]
    term = [(o, remap[a]) for o, a in gens if o == terminal][:1]
    rest = [g for g in gens if g[0] not in (GEN_KEY_RANGE, GEN_VEL_RANGE, terminal)]
    return kr + vr + rest + term


def _write_sf2(out_path: str, sf2: SF2Parsed, t: SF2Trimmed) -> None:
    def hydra(entries, rec_len: int, bag_field: int, terminal_gen: int, remap, terminal_name: bytes):
        hdr, bag, mod, gen = bytearray(), bytearray(), bytearray(), bytearray()
        n_gen = n_mod = n_bag = 0
        for h, zones in entries:
            rec = bytearray(h.raw)
            struct.pack_into("<H", rec, bag_field, n_bag)
            hdr += rec
            for z in zones:
                bag += struct.pack("<HH", n_gen, n_mod)
                n_bag += 1
                for o, a in _ordered_gens(z.gens, terminal_gen, remap):
                    gen += struct.pack("<HH", o, a & 0xFFFF)
                    n_gen += 1
                for m in z.mods:
                    mod += m
                    n_mod += 1
        term = bytearray(rec_len)
        term[:20] = terminal_name.ljust(20, b"\x00")
        struct.pack_into("<H", term, bag_field, n_bag)
        hdr += term
        bag += struct.pack("<HH", n_gen, n_mod)
        mod += b"\x00" * 10
        gen += b"\x00" * 4
        return bytes(hdr), bytes(bag), bytes(mod), bytes(gen)

    phdr, pbag, pmod, pgen = hydra(t.presets, 38, 24, GEN_INSTRUMENT, t.inst_map, b"EOP")
    inst, ibag, imod, igen = hydra(t.insts, 22, 20, GEN_SAMPLE_ID, t.sample_map, b"EOI")

    compressed = any(sf2.samples[i].stype & SAMPLE_COMPRESSED for i in t.sample_list)
    smpl = bytearray()
    sm24 = bytearray()
    use_sm24 = sf2.sm24 is not None and not compressed
    shdr = bytearray()
    for old in t.sample_list:
        s = sf2.samples[old]
        rec = bytearray(s.raw)
        stype = s.stype
        if not stype & (0x0F | SAMPLE_COMPRESSED):
            stype |= 1  # invalid type 0 on a real sample: mono (avoids "ROM sample" prompts)
        link = s.link
        if s.stype & SAMPLE_LINK_TYPES:
            if s.link in t.sample_map:
                link = t.sample_map[s.link]
            else:
                link, stype = 0, (stype & ~SAMPLE_LINK_TYPES) | 1  # partner missing: make it mono
        if s.stype & SAMPLE_ROM:
            pass  # ROM sample data is not in smpl; keep its offsets
        elif s.stype & SAMPLE_COMPRESSED:
            # SF3: start/end are byte offsets of the Ogg blob; loop points are relative and kept as-is.
            blob = sf2.smpl[s.start: s.end]
            struct.pack_into("<II", rec, 20, len(smpl), len(smpl) + len(blob))
            smpl += blob + b"\x00" * (len(blob) & 1)
        else:
            length = max(0, s.end - s.start)
            pos = len(smpl) // 2
            smpl += sf2.smpl[s.start * 2: s.end * 2].ljust(length * 2, b"\x00") + b"\x00" * (SAMPLE_PAD * 2)
            if use_sm24:
                sm24 += sf2.sm24[s.start: s.end].ljust(length, b"\x00") + b"\x00" * SAMPLE_PAD  # type: ignore[index]
            ls, le = s.start_loop - s.start, s.end_loop - s.start
            if not (0 <= ls < le <= length):
                ls, le = 0, length  # invalid/absent loop (e.g. 0/0 on unlooped rips): span whole sample, like Awave
            struct.pack_into("<IIII", rec, 20, pos, pos + length, pos + ls, pos + le)
        struct.pack_into("<HH", rec, 42, link, stype)
        shdr += rec
    shdr += b"EOS".ljust(46, b"\x00")

    info = b"".join(_riff_chunk(k, v) for k, v in sf2.info) or _riff_chunk("ifil", struct.pack("<HH", 2, 1))
    sdta = _riff_chunk("smpl", bytes(smpl))
    if use_sm24:
        sdta += _riff_chunk("sm24", bytes(sm24))
    pdta = b"".join(_riff_chunk(n, d) for n, d in (
        ("phdr", phdr), ("pbag", pbag), ("pmod", pmod), ("pgen", pgen),
        ("inst", inst), ("ibag", ibag), ("imod", imod), ("igen", igen), ("shdr", bytes(shdr)),
    ))
    body = _riff_list("INFO", info) + _riff_list("sdta", sdta) + _riff_list("pdta", pdta)
    with open(out_path, "wb") as f:
        f.write(_riff_chunk("RIFF", b"sfbk" + body))


def _sf2_report(t: SF2Trimmed, sf2: SF2Parsed) -> str:
    lines = []
    for h, zones in t.presets:
        prog, bank = struct.unpack_from("<HH", h.raw, 20)
        lines.append(f"  preset bank {bank:3d} prog {prog:3d} '{_read_zstr(h.raw[:20])}' zones {len(zones)}")
    lines.append(f"Keeping {len(t.presets)} of {len(sf2.presets)} presets, {len(t.insts)} of {len(sf2.insts)} "
                 f"instruments, {len(t.sample_list)} of {len(sf2.samples)} samples.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DLS (level 1 and 2)
# Every chunk that does not need changing (instrument/region articulation, DLS1 'rgn '/
# 'art1'/'lart', dlid, vers, INFO, wave wsmp, ...) is copied byte for byte. Only colh,
# insh region counts, invalid rgnh ranges, wlnk wave indices, ptbl and broken PCM fmt
# fields are rewritten.
# ---------------------------------------------------------------------------

DLS_DRUM_FLAG = 0x80000000


@dataclass
class DLSRegion:
    chunk: Chunk                 # LIST 'rgn ' or 'rgn2'
    key: Tuple[int, int]
    vel: Tuple[int, int]
    table_index: Optional[int]   # ptbl cue index from wlnk


@dataclass
class DLSInstrument:
    chunk: Chunk                 # LIST 'ins '
    bank: int
    program: int
    name: str
    regions: List[DLSRegion]


@dataclass
class DLSParsed:
    form: str
    top: List[Chunk]
    instruments: List[DLSInstrument]
    waves: List[Optional[Chunk]]  # indexed by ptbl cue index


def _info_name(c: Chunk, default: str) -> str:
    for info in c.children or []:
        if _is_list(info, "INFO"):
            inam = _find_child(info, "INAM")
            if inam and len(inam.data):
                return _read_zstr(inam.data.tobytes())
    return default


def _parse_dls(path: str) -> DLSParsed:
    with open(path, "rb") as f:
        root = _parse_riff(f.read())
    if root.list_type not in ("DLS ", "MLID"):
        raise RiffError("Not a DLS (RIFF DLS )")
    top = root.children or []
    lins = next((c for c in top if _is_list(c, "lins")), None)
    wvpl = next((c for c in top if _is_list(c, "wvpl")), None)
    ptbl = next((c for c in top if c.fourcc == "ptbl"), None)
    if not lins or not wvpl:
        raise RiffError("Malformed DLS: missing lins/wvpl")

    # ptbl offsets are relative to the first byte after the 'wvpl' list type.
    by_offset: Dict[int, Chunk] = {}
    off = 0
    for c in wvpl.children or []:
        by_offset[off] = c
        off += 8 + len(c.data) + (len(c.data) & 1)
    waves: List[Optional[Chunk]]
    if ptbl and len(ptbl.data) >= 8:
        cb, n = struct.unpack_from("<II", ptbl.data, 0)
        offsets = [_read_u32le(ptbl.data, cb + 4 * i) for i in range(n) if cb + 4 * i + 4 <= len(ptbl.data)]
        waves = [by_offset.get(o) for o in offsets]
    else:
        waves = list(wvpl.children or [])

    instruments: List[DLSInstrument] = []
    for c in lins.children or []:
        if not _is_list(c, "ins "):
            continue
        insh = _find_child(c, "insh")
        if not insh or len(insh.data) < 12:
            continue
        lrgn = next((x for x in c.children or [] if _is_list(x, "lrgn")), None)
        regions: List[DLSRegion] = []
        for r in (lrgn.children if lrgn else None) or []:
            if not (_is_list(r, "rgn ") or _is_list(r, "rgn2")):
                continue
            rgnh = _find_child(r, "rgnh")
            wlnk = _find_child(r, "wlnk")
            if rgnh and len(rgnh.data) >= 8:
                klo, khi, vlo, vhi = struct.unpack_from("<HHHH", rgnh.data, 0)
            else:
                klo, khi, vlo, vhi = 0, 127, 0, 127
            if r.list_type == "rgn " or vhi == 0:
                vlo, vhi = 0, 127  # DLS level 1 ignores velocity ranges (often left at 0-0)
            ti = _read_u32le(wlnk.data, 8) if wlnk and len(wlnk.data) >= 12 else None
            regions.append(DLSRegion(r, (klo, khi), (vlo, vhi), ti))
        instruments.append(DLSInstrument(
            chunk=c,
            bank=_read_u32le(insh.data, 4),
            program=_read_u32le(insh.data, 8),
            name=_info_name(c, "Instrument"),
            regions=regions,
        ))
    return DLSParsed(root.list_type or "DLS ", top, instruments, waves)


def _dls_key(instr: DLSInstrument) -> Tuple[bool, int, int, int]:
    b = instr.bank
    return bool(b & DLS_DRUM_FLAG), (b >> 8) & 0x7F, b & 0x7F, instr.program & 0x7F


def _dls_trim(
    dls: DLSParsed, usage: Usage, style: str
) -> Tuple[List[Tuple[DLSInstrument, List[DLSRegion]]], List[int], List[str]]:
    lookup: Dict[Tuple[bool, int, int, int], List[int]] = {}
    for i, instr in enumerate(dls.instruments):
        lookup.setdefault(_dls_key(instr), []).append(i)
    has_drums = any(k[0] for k in lookup)

    def candidates(sel: Sel) -> List[List[Hashable]]:
        m, l, p = sel.msb, sel.lsb, sel.prog
        if sel.drum and has_drums:
            return [[(True, m, l, p), (True, 0, 0, p), (True, 0, 0, 0)]]
        banks = [(0, 0)] if style == "none" else [(m, l)]
        if style == "all" or (style == "auto" and sel.mode == "gm"):
            banks.append((0, 0))  # GM devices ignore bank select
        return [[(False, bm, bl, p), (False, bm, 0, p), (False, 0, 0, p), (False, 0, 0, 0)]
                for bm, bl in dict.fromkeys(banks)]

    def describe(k) -> str:
        return f"{'drum ' if k[0] else ''}bank {k[1]}/{k[2]} program {k[3]}"

    prog_notes, msgs = resolve_usage(usage, set(lookup), candidates, describe)

    kept: List[Tuple[DLSInstrument, List[DLSRegion]]] = []
    used_waves: Set[int] = set()
    selected = sorted({i for k in prog_notes for i in lookup[k]})  # type: ignore[index]
    for idx in selected:
        instr = dls.instruments[idx]
        notes = prog_notes[_dls_key(instr)]
        regions = [
            r for r in instr.regions
            if r.table_index is not None and r.table_index < len(dls.waves) and dls.waves[r.table_index] is not None
            and any(r.key[0] <= n <= r.key[1] and r.vel[0] <= v <= r.vel[1] for n, v in notes)
        ]
        if regions:
            kept.append((instr, regions))
            used_waves.update(r.table_index for r in regions)  # type: ignore[misc]
    if not kept:
        raise ValueError("\n".join(["No instrument in the DLS is played by the MIDI file."] + msgs))
    return kept, sorted(used_waves), msgs


def _dls_region_bytes(r: DLSRegion, wave_map: Dict[int, int]) -> bytes:
    body = bytearray(r.chunk.data[:4].tobytes())  # list type
    for c in r.chunk.children or []:
        if c.fourcc == "rgnh" and len(c.data) >= 8:
            d = bytearray(c.data.tobytes())
            for o in (0, 2, 4, 6):  # ranges above 127 are invalid (some converters write them)
                struct.pack_into("<H", d, o, min(_read_u16le(d, o), 127))
            body += _riff_chunk("rgnh", bytes(d))
        elif c.fourcc == "wlnk" and len(c.data) >= 12:
            d = bytearray(c.data.tobytes())
            struct.pack_into("<I", d, 8, wave_map[r.table_index])  # type: ignore[index]
            body += _riff_chunk("wlnk", bytes(d))
        else:
            body += _raw(c)
    return _riff_chunk("LIST", bytes(body))


def _dls_wave_bytes(w: Chunk) -> bytes:
    if w.fourcc != "LIST":
        return _raw(w)
    body = bytearray(w.data[:4].tobytes())
    for c in w.children or []:
        if c.fourcc == "fmt " and len(c.data) >= 16:
            d = bytearray(c.data.tobytes())
            tag, chans, rate, _, _, bits = struct.unpack_from("<HHIIHH", d, 0)
            if tag == 1 and chans and bits:  # PCM: derive nAvgBytesPerSec / nBlockAlign (often wrong in rips)
                align = chans * ((bits + 7) // 8)
                struct.pack_into("<IH", d, 8, rate * align, align)
            body += _riff_chunk("fmt ", bytes(d))
        else:
            body += _raw(c)
    return _riff_chunk("LIST", bytes(body))


def _write_dls(
    out_path: str, dls: DLSParsed, kept: List[Tuple[DLSInstrument, List[DLSRegion]]], wave_list: List[int]
) -> None:
    wave_map = {old: new for new, old in enumerate(wave_list)}

    lins_body = bytearray(b"lins")
    for instr, regions in kept:
        lrgn = _riff_list("lrgn", b"".join(_dls_region_bytes(r, wave_map) for r in regions))
        body = bytearray(b"ins ")
        wrote_lrgn = False
        for c in instr.chunk.children or []:
            if c.fourcc == "insh":
                d = bytearray(c.data.tobytes())
                struct.pack_into("<I", d, 0, len(regions))
                body += _riff_chunk("insh", bytes(d))
            elif _is_list(c, "lrgn"):
                body += lrgn
                wrote_lrgn = True
            else:
                body += _raw(c)
        if not wrote_lrgn:
            body += lrgn
        lins_body += _riff_chunk("LIST", bytes(body))

    wvpl_body = bytearray(b"wvpl")
    offsets = []
    for wi in wave_list:
        offsets.append(len(wvpl_body) - 4)
        wvpl_body += _dls_wave_bytes(dls.waves[wi])  # type: ignore[arg-type]
    ptbl = _riff_chunk("ptbl", struct.pack("<II", 8, len(offsets)) + b"".join(struct.pack("<I", o) for o in offsets))
    colh = _riff_chunk("colh", struct.pack("<I", len(kept)))

    parts: List[bytes] = []
    if not any(c.fourcc == "colh" for c in dls.top):
        parts.append(colh)
    has_ptbl = any(c.fourcc == "ptbl" for c in dls.top)
    for c in dls.top:
        if c.fourcc == "colh":
            parts.append(colh)
        elif _is_list(c, "lins"):
            parts.append(_riff_chunk("LIST", bytes(lins_body)))
        elif c.fourcc == "ptbl":
            parts.append(ptbl)
        elif _is_list(c, "wvpl"):
            if not has_ptbl:
                parts.append(ptbl)
            parts.append(_riff_chunk("LIST", bytes(wvpl_body)))
        else:
            parts.append(_raw(c))
    with open(out_path, "wb") as f:
        f.write(_riff_chunk("RIFF", dls.form.encode("ascii")[:4] + b"".join(parts)))


def _dls_report(kept: List[Tuple[DLSInstrument, List[DLSRegion]]], dls: DLSParsed, wave_list: List[int]) -> str:
    lines = []
    for instr, regions in kept:
        drum, m, l, p = _dls_key(instr)
        lines.append(f"  instrument bank {m:3d}/{l:3d}{' drum' if drum else ''} prog {p:3d} "
                     f"'{instr.name}' regions {len(regions)}/{len(instr.regions)}")
    lines.append(f"Keeping {len(kept)} of {len(dls.instruments)} instruments, "
                 f"{len(wave_list)} of {len(dls.waves)} waves.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BANK_STYLES = ("auto", "msb", "gs", "gm", "lsb", "mma", "xg", "none", "all")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Trim SF2/SF3/DLS to the presets, zones and samples a MIDI file plays.",
        epilog="See the module docstring for details on MIDI and bank-select handling.",
    )
    ap.add_argument("midi", help="Input MIDI file (.mid / .midi / .rmi / .kar)")
    ap.add_argument("soundfont", help="Input SF2, SF3 or DLS file (detected by content, any extension)")
    ap.add_argument("-o", "--output", required=True, help="Output file")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be kept, do not write")
    ap.add_argument("--bank-style", choices=BANK_STYLES, default="auto",
                    help="How bank select maps to soundfont banks (default: auto, follows GS/XG/GM2 sysex)")
    args = ap.parse_args()
    style = "msb" if args.bank_style in ("gs", "gm") else args.bank_style

    try:
        usage = collect_midi_usage(args.midi, style)
    except (MidiError, OSError) as e:
        print(f"Error reading MIDI: {e}", file=sys.stderr)
        return 1
    if not usage:
        print("Error: the MIDI file plays no notes.", file=sys.stderr)
        return 1

    try:
        with open(args.soundfont, "rb") as f:
            head = f.read(12)
        form = head[8:12].decode("ascii", "replace") if head[:4] == b"RIFF" else ""
        if form == "sfbk":
            sf2 = _parse_sf2(args.soundfont)
            trimmed, msgs = _sf2_trim(sf2, usage, style)
            msgs.append(_sf2_report(trimmed, sf2))
            if not args.dry_run:
                _write_sf2(args.output, sf2, trimmed)
        elif form in ("DLS ", "MLID"):
            dls = _parse_dls(args.soundfont)
            kept, wave_list, msgs = _dls_trim(dls, usage, style)
            msgs.append(_dls_report(kept, dls, wave_list))
            if not args.dry_run:
                _write_dls(args.output, dls, kept, wave_list)
        else:
            print("Error: not an SF2/SF3 or DLS file (RIFF form must be sfbk or DLS ).", file=sys.stderr)
            return 1
    except (ValueError, RiffError, OSError, struct.error) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for m in msgs:
        print(m)
    if not args.dry_run:
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
