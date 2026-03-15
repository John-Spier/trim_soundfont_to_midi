#!/usr/bin/env python3
"""
Trim an SF2 or DLS file to only include presets/instruments and samples/waves
that are referenced by a specific MIDI file (by Program Change and Bank Select).

Usage:
  python trim_soundfont_to_midi.py midi.mid soundfont.sf2 -o out.sf2
  python trim_soundfont_to_midi.py midi.mid collection.dls -o out.dls
  python trim_soundfont_to_midi.py midi.mid soundfont.sf2 -o out.sf2 --dry-run
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Optional: mido for MIDI
try:
    import mido
except ImportError:
    mido = None  # type: ignore


# --- Bank normalization ---

def normalize_bank_mma(bank: int) -> int:
    """14-bit: (MSB<<8)|LSB, mask to 0x7F7F."""
    return bank & 0x7F7F


def normalize_bank_gm(bank: int) -> int:
    """7-bit: only MSB."""
    return (bank >> 8) & 0x7F


def normalize_bank(bank: int, style: str) -> int:
    if style == "gm":
        return normalize_bank_gm(bank)
    return normalize_bank_mma(bank)


# --- MIDI: collect (bank, program) used ---

def collect_midi_bank_program(midi_path: str, bank_style: str = "mma") -> Set[Tuple[int, int]]:
    """Parse MIDI and return set of (normalized_bank, program) used."""
    if mido is None:
        raise RuntimeError("mido is required for MIDI parsing. Install with: pip install mido")
    used: Set[Tuple[int, int]] = set()
    mid = mido.MidiFile(midi_path)
    # Per-channel state: bank_msb, bank_lsb (0..15)
    bank_msb: List[int] = [0] * 16
    bank_lsb: List[int] = [0] * 16

    for track in mid.tracks:
        for msg in track:
            if not hasattr(msg, "channel"):
                continue
            ch = getattr(msg, "channel", None)
            if ch is None or not (0 <= ch <= 15):
                continue
            if msg.type == "control_change":
                if msg.control == 0:
                    bank_msb[ch] = msg.value & 0x7F
                elif msg.control == 32:
                    bank_lsb[ch] = msg.value & 0x7F
            elif msg.type == "program_change":
                bank_raw = (bank_msb[ch] << 8) | bank_lsb[ch]
                nbank = normalize_bank(bank_raw, bank_style)
                used.add((nbank, msg.program & 0x7F))
    return used


# --- RIFF helpers (minimal, no dependency on dls_sf2_to_segsat) ---

class RiffError(Exception):
    pass


@dataclass
class Chunk:
    fourcc: str
    data: memoryview
    children: Optional[List[Chunk]] = None
    list_type: Optional[str] = None


def _read_u32le(buf: memoryview, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _read_u16le(buf: memoryview, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def _read_s16le(buf: memoryview, off: int) -> int:
    return struct.unpack_from("<h", buf, off)[0]


def _parse_riff(buf: bytes) -> Chunk:
    mv = memoryview(buf)
    if len(mv) < 12:
        raise RiffError("File too small for RIFF")
    if mv[0:4].tobytes() != b"RIFF":
        raise RiffError("Not a RIFF file")
    size = _read_u32le(mv, 4)
    form = mv[8:12].tobytes().decode("ascii", "replace")
    root_data = mv[12 : 8 + size]
    root = Chunk("RIFF", root_data, children=[], list_type=form)
    root.children = _parse_children(root_data, 0, len(root_data))
    return root


def _parse_children(mv: memoryview, start: int, end: int) -> List[Chunk]:
    out: List[Chunk] = []
    off = start
    while off + 8 <= end:
        fourcc = mv[off : off + 4].tobytes().decode("ascii", "replace")
        size = _read_u32le(mv, off + 4)
        data_start = off + 8
        data_end = data_start + size
        if data_end > end:
            break
        data = mv[data_start:data_end]
        ch = Chunk(fourcc, data, children=None, list_type=None)
        if fourcc in ("RIFF", "LIST"):
            if len(data) < 4:
                raise RiffError(f"{fourcc} chunk too small")
            list_type = data[0:4].tobytes().decode("ascii", "replace")
            ch.list_type = list_type
            ch.children = _parse_children(data, 4, len(data))
        out.append(ch)
        off = data_end + (size & 1)
    return out


def _find_list(root: Chunk, list_type: str) -> Optional[Chunk]:
    if root.fourcc in ("RIFF", "LIST") and root.list_type == list_type:
        return root
    if not root.children:
        return None
    for c in root.children:
        got = _find_list(c, list_type)
        if got:
            return got
    return None


def _find_child(parent: Chunk, fourcc: str) -> Optional[Chunk]:
    if not parent.children:
        return None
    for c in parent.children:
        if c.fourcc == fourcc:
            return c
    return None


def _read_zstr(b: bytes) -> str:
    s = b.split(b"\x00", 1)[0]
    return s.decode("ascii", "replace")


# --- SF2 generator IDs ---
SF2_GEN_INSTRUMENT = 41
SF2_GEN_SAMPLE_ID = 53


def _sf2_normalize_bank(bank: int, bank_style: str) -> int:
    """SF2 wBank is 16-bit; often 0-127. Normalize for comparison."""
    if bank_style == "gm":
        return bank & 0x7F
    return bank & 0x7F7F


# --- SF2: read, filter, write ---

@dataclass
class SF2Parsed:
    phdr: List[Tuple[str, int, int, int]]  # name, preset, bank, pbag_ndx
    pbag: List[Tuple[int, int]]
    pgen: List[Tuple[int, int]]
    inst: List[Tuple[str, int]]  # name, ibag_ndx
    ibag: List[Tuple[int, int]]
    igen: List[Tuple[int, int]]
    shdr: List[Dict]
    smpl_bytes: bytes
    pdta_chunks_order: List[str]  # e.g. phdr, pbag, pmod, pgen, inst, ibag, imod, igen, shdr
    info_chunks: Dict[str, bytes]  # ifil, isng, INAM, etc. from INFO
    raw_pmod: bytes
    raw_imod: bytes


def _parse_sf2_raw(path: str) -> SF2Parsed:
    data = open(path, "rb").read()
    root = _parse_riff(data)
    if root.list_type != "sfbk":
        raise RiffError("Not an SF2 (RIFF sfbk)")

    sdta = _find_list(root, "sdta")
    pdta = _find_list(root, "pdta")
    if not sdta or not pdta:
        raise RiffError("Malformed SF2: missing sdta/pdta")

    smpl_chunk = _find_child(sdta, "smpl")
    if not smpl_chunk:
        raise RiffError("Malformed SF2: missing sdta/smpl")
    smpl_bytes = smpl_chunk.data.tobytes()

    def need(name: str) -> Chunk:
        c = _find_child(pdta, name)
        if not c:
            raise RiffError(f"Malformed SF2: missing pdta/{name}")
        return c

    phdr_d = need("phdr").data
    pbag_d = need("pbag").data
    pgen_d = need("pgen").data
    inst_d = need("inst").data
    ibag_d = need("ibag").data
    igen_d = need("igen").data
    shdr_d = need("shdr").data
    pmod_chunk = _find_child(pdta, "pmod")
    imod_chunk = _find_child(pdta, "imod")
    raw_pmod = pmod_chunk.data.tobytes() if pmod_chunk else b""
    raw_imod = imod_chunk.data.tobytes() if imod_chunk else b""

    rec_sz = 38
    phdr_list: List[Tuple[str, int, int, int]] = []
    for off in range(0, len(phdr_d), rec_sz):
        if off + rec_sz > len(phdr_d):
            break
        name = _read_zstr(phdr_d[off : off + 20].tobytes())
        preset = _read_u16le(phdr_d, off + 20)
        bank = _read_u16le(phdr_d, off + 22)
        pbag_ndx = _read_u16le(phdr_d, off + 24)
        phdr_list.append((name, preset, bank, pbag_ndx))

    rec_sz = 22
    inst_list: List[Tuple[str, int]] = []
    for off in range(0, len(inst_d), rec_sz):
        if off + rec_sz > len(inst_d):
            break
        name = _read_zstr(inst_d[off : off + 20].tobytes())
        ibag_ndx = _read_u16le(inst_d, off + 20)
        inst_list.append((name, ibag_ndx))

    rec_sz = 46
    shdr_list: List[Dict] = []
    for off in range(0, len(shdr_d), rec_sz):
        if off + rec_sz > len(shdr_d):
            break
        name = _read_zstr(shdr_d[off : off + 20].tobytes())
        start = _read_u32le(shdr_d, off + 20)
        end = _read_u32le(shdr_d, off + 24)
        start_loop = _read_u32le(shdr_d, off + 28)
        end_loop = _read_u32le(shdr_d, off + 32)
        rate = _read_u32le(shdr_d, off + 36)
        orig_pitch = int(shdr_d[off + 40])
        pitch_corr = struct.unpack_from("<b", shdr_d, off + 41)[0]
        shdr_list.append({
            "name": name, "start": start, "end": end,
            "start_loop": start_loop, "end_loop": end_loop, "rate": rate,
            "orig_pitch": orig_pitch, "pitch_corr": pitch_corr,
            "raw": shdr_d[off : off + rec_sz].tobytes(),
        })

    def parse_bag(bag_d: memoryview) -> List[Tuple[int, int]]:
        out = []
        for off in range(0, len(bag_d), 4):
            if off + 4 > len(bag_d):
                break
            out.append((_read_u16le(bag_d, off), _read_u16le(bag_d, off + 2)))
        return out

    def parse_gen(gen_d: memoryview) -> List[Tuple[int, int]]:
        out = []
        for off in range(0, len(gen_d), 4):
            if off + 4 > len(gen_d):
                break
            out.append((_read_u16le(gen_d, off), _read_s16le(gen_d, off + 2)))
        return out

    pbag_list = parse_bag(pbag_d)
    pgen_list = parse_gen(pgen_d)
    ibag_list = parse_bag(ibag_d)
    igen_list = parse_gen(igen_d)

    info_chunks: Dict[str, bytes] = {}
    info_list = _find_list(root, "INFO")
    if info_list and info_list.children:
        for c in info_list.children:
            if c.fourcc not in ("RIFF", "LIST") and len(c.data) >= 4:
                info_chunks[c.fourcc] = c.data.tobytes()

    return SF2Parsed(
        phdr=phdr_list,
        pbag=pbag_list,
        pgen=pgen_list,
        inst=inst_list,
        ibag=ibag_list,
        igen=igen_list,
        shdr=shdr_list,
        smpl_bytes=smpl_bytes,
        pdta_chunks_order=["phdr", "pbag", "pmod", "pgen", "inst", "ibag", "imod", "igen", "shdr"],
        info_chunks=info_chunks,
        raw_pmod=raw_pmod,
        raw_imod=raw_imod,
    )


def _bag_gens(gen_list: List[Tuple[int, int]], bag_ranges: List[Tuple[int, int]], bag_idx: int) -> Dict[int, int]:
    if bag_idx < 0 or bag_idx >= len(bag_ranges):
        return {}
    gen_start = bag_ranges[bag_idx][0]
    gen_end = bag_ranges[bag_idx + 1][0] if bag_idx + 1 < len(bag_ranges) else len(gen_list)
    return dict(gen_list[gen_start:gen_end])


def _sf2_filter_and_remap(
    sf2: SF2Parsed,
    used_bank_program: Set[Tuple[int, int]],
    bank_style: str,
) -> Tuple[
    List[Tuple[str, int, int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[str, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Dict],
    bytes,
    Dict[int, int],
    Dict[int, int],
]:
    """Filter to used presets/instruments/samples; return new lists and sample data plus old->new index maps."""
    used_preset_indices: Set[int] = set()
    for i, (name, preset_num, bank_num, pbag_ndx) in enumerate(sf2.phdr):
        if i >= len(sf2.phdr) - 1:
            continue  # skip terminal
        nb = _sf2_normalize_bank(bank_num, bank_style)
        if (nb, preset_num) in used_bank_program:
            used_preset_indices.add(i)

    if not used_preset_indices:
        raise ValueError("No presets in the soundfont match the (bank, program) used in the MIDI file.")

    used_inst_indices: Set[int] = set()
    for p_idx in used_preset_indices:
        _, _, _, pbag_ndx = sf2.phdr[p_idx]
        next_pbag = sf2.phdr[p_idx + 1][3]
        for bag_i in range(pbag_ndx, next_pbag):
            gens = _bag_gens(sf2.pgen, sf2.pbag, bag_i)
            inst_id = gens.get(SF2_GEN_INSTRUMENT)
            if inst_id is not None and 0 <= inst_id < len(sf2.inst) - 1:
                used_inst_indices.add(inst_id)

    used_sample_indices: Set[int] = set()
    for inst_idx in used_inst_indices:
        ibag_ndx = sf2.inst[inst_idx][1]
        next_ibag = sf2.inst[inst_idx + 1][1]
        for ib in range(ibag_ndx, next_ibag):
            gens = _bag_gens(sf2.igen, sf2.ibag, ib)
            samp_id = gens.get(SF2_GEN_SAMPLE_ID)
            if samp_id is not None and 0 <= samp_id < len(sf2.shdr):
                used_sample_indices.add(samp_id)

    # Include terminal sample (last shdr) if present
    if len(sf2.shdr) > 0:
        used_sample_indices.add(len(sf2.shdr) - 1)

    used_preset_list = sorted(used_preset_indices)
    used_inst_list = sorted(used_inst_indices)
    used_sample_list = sorted(used_sample_indices)

    old_to_new_inst: Dict[int, int] = {old: new for new, old in enumerate(used_inst_list)}
    old_to_new_sample: Dict[int, int] = {old: new for new, old in enumerate(used_sample_list)}

    # Build new phdr, pbag, pgen (presets only)
    new_phdr: List[Tuple[str, int, int, int]] = []
    new_pbag: List[Tuple[int, int]] = []
    new_pgen: List[Tuple[int, int]] = []

    pbag_count = 0
    pgen_count = 0
    for new_p_idx, old_p_idx in enumerate(used_preset_list):
        name, preset_num, bank_num, pbag_ndx = sf2.phdr[old_p_idx]
        next_pbag = sf2.phdr[old_p_idx + 1][3]
        new_phdr.append((name, preset_num, bank_num, pbag_count))
        for bag_i in range(pbag_ndx, next_pbag):
            gens = _bag_gens(sf2.pgen, sf2.pbag, bag_i)
            inst_id = gens.get(SF2_GEN_INSTRUMENT)
            if inst_id is not None and inst_id in old_to_new_inst:
                new_inst_id = old_to_new_inst[inst_id]
                new_pbag.append((pgen_count, 0))
                new_pgen.append((56, 0))   # reverbEffectsSend 0
                new_pgen.append((SF2_GEN_INSTRUMENT, new_inst_id))
                pgen_count += 2
                pbag_count += 1
    new_phdr.append(("EOP", 0, 0, pbag_count))
    new_pbag.append((pgen_count, 0))  # terminal pbag

    # Build new inst, ibag, igen (instruments only)
    new_inst: List[Tuple[str, int]] = []
    new_ibag: List[Tuple[int, int]] = []
    new_igen: List[Tuple[int, int]] = []

    ibag_count = 0
    igen_count = 0
    for new_i_idx, old_i_idx in enumerate(used_inst_list):
        name, ibag_ndx = sf2.inst[old_i_idx]
        next_ibag = sf2.inst[old_i_idx + 1][1]
        new_inst.append((name, ibag_count))
        for ib in range(ibag_ndx, next_ibag):
            gen_start = sf2.ibag[ib][0]
            gen_end = sf2.ibag[ib + 1][0] if ib + 1 < len(sf2.ibag) else len(sf2.igen)
            new_ibag.append((igen_count, 0))
            for oper, amt in sf2.igen[gen_start:gen_end]:
                if oper == SF2_GEN_SAMPLE_ID and amt in old_to_new_sample:
                    new_igen.append((oper, old_to_new_sample[amt]))
                else:
                    new_igen.append((oper, amt))
            igen_count += gen_end - gen_start
        ibag_count += next_ibag - ibag_ndx
    new_inst.append(("EOI", ibag_count))
    new_ibag.append((igen_count, 0))

    # Build new shdr and smpl
    new_shdr: List[Dict] = []
    smpl_parts: List[bytes] = []
    SF2_PAD = 46 * 2  # 46 padding samples per SF2 spec
    offset = 0
    for new_s_idx, old_s_idx in enumerate(used_sample_list):
        h = sf2.shdr[old_s_idx].copy()
        start = int(h["start"])
        end = int(h["end"])
        start_loop = int(h["start_loop"])
        end_loop = int(h["end_loop"])
        length_words = end - start
        if length_words <= 0 and old_s_idx < len(sf2.shdr) - 1:
            continue
        if old_s_idx == len(sf2.shdr) - 1:
            new_shdr.append({
                **h,
                "start": offset,
                "end": offset,
                "start_loop": offset,
                "end_loop": offset,
            })
            break
        sample_data = sf2.smpl_bytes[start * 2 : end * 2]
        smpl_parts.append(sample_data)
        smpl_parts.append(b"\x00" * SF2_PAD)
        new_shdr.append({
            **h,
            "start": offset,
            "end": offset + length_words,
            "start_loop": offset + max(0, start_loop - start),
            "end_loop": offset + max(0, end_loop - start),
        })
        offset += length_words + 46
    new_smpl = b"".join(smpl_parts)

    # Ensure terminal EOS sample header exists
    if not new_shdr or new_shdr[-1].get("end") != new_shdr[-1].get("start"):
        new_shdr.append({
            "name": "EOS", "start": offset, "end": offset, "start_loop": offset, "end_loop": offset,
            "rate": 0, "orig_pitch": 0, "pitch_corr": 0,
        })

    return (
        new_phdr, new_pbag, new_pgen,
        new_inst, new_ibag, new_igen,
        new_shdr,
        new_smpl,
        old_to_new_inst,
        old_to_new_sample,
    )


def _write_sf2(
    out_path: str,
    sf2: SF2Parsed,
    new_phdr: List[Tuple[str, int, int, int]],
    new_pbag: List[Tuple[int, int]],
    new_pgen: List[Tuple[int, int]],
    new_inst: List[Tuple[str, int]],
    new_ibag: List[Tuple[int, int]],
    new_igen: List[Tuple[int, int]],
    new_shdr: List[Dict],
    new_smpl: bytes,
) -> None:
    def w_u16le(v: int) -> bytes:
        return struct.pack("<H", v & 0xFFFF)

    def w_s16le(v: int) -> bytes:
        return struct.pack("<h", v)

    def w_u32le(v: int) -> bytes:
        return struct.pack("<I", v & 0xFFFFFFFF)

    def chunk(fourcc: str, data: bytes) -> bytes:
        pad = len(data) & 1  # RIFF: chunk data padded to 2-byte boundary
        return fourcc.encode("ascii") + w_u32le(len(data)) + data + (b"\x00" * pad)

    # INFO list (each chunk: fourcc + size + payload)
    info_parts = []
    for k, v in sf2.info_chunks.items():
        info_parts.append(k.encode("ascii")[:4] + w_u32le(len(v)) + v)
    info_data = b"".join(info_parts)
    if not info_data:
        info_data = b"ifil\x04\x00\x00\x00\x02\x00\x01\x00"  # version 2.1
    info_list = b"LIST" + w_u32le(4 + len(info_data)) + b"INFO" + info_data
    if len(info_list) & 1:
        info_list += b"\x00"

    # sdta list (smpl chunk data must be word-aligned per RIFF)
    smpl_chunk = b"smpl" + w_u32le(len(new_smpl)) + new_smpl
    if len(new_smpl) & 1:
        smpl_chunk += b"\x00"
    sdta_list = b"LIST" + w_u32le(4 + len(smpl_chunk)) + b"sdta" + smpl_chunk
    if len(sdta_list) & 1:
        sdta_list += b"\x00"

    # pdta: phdr, pbag, pmod, pgen, inst, ibag, imod, igen, shdr
    phdr_data = b""
    for name, preset, bank, pbag_ndx in new_phdr:
        phdr_data += (name[:20].encode("ascii") + b"\x00" * 20)[:20]
        phdr_data += w_u16le(preset)
        phdr_data += w_u16le(bank)
        phdr_data += w_u16le(pbag_ndx)
        phdr_data += w_u32le(0) * 3
    pbag_data = b""
    for g, m in new_pbag:
        pbag_data += w_u16le(g) + w_u16le(m)
    pmod_data = sf2.raw_pmod if sf2.raw_pmod else (w_u16le(0) + w_u16le(0) + w_u16le(0) + w_u16le(0))
    pgen_data = b""
    for op, amt in new_pgen:
        pgen_data += w_u16le(op) + w_s16le(amt)
    inst_data = b""
    for name, ibag_ndx in new_inst:
        inst_data += (name[:20].encode("ascii") + b"\x00" * 20)[:20]
        inst_data += w_u16le(ibag_ndx)
    ibag_data = b""
    for g, m in new_ibag:
        ibag_data += w_u16le(g) + w_u16le(m)
    imod_data = sf2.raw_imod if sf2.raw_imod else (w_u16le(0) + w_u16le(0) + w_u16le(0) + w_u16le(0))
    igen_data = b""
    for op, amt in new_igen:
        igen_data += w_u16le(op) + w_s16le(amt)
    # shdr: 46 bytes per record (sfSample). Last field is sfSampleType: 1=mono, 0=EOS (no ROM/linked)
    shdr_data = b""
    for h in new_shdr:
        name = (h.get("name", "") or "EOS")[:20]
        shdr_data += (name.encode("ascii") + b"\x00" * 20)[:20]
        shdr_data += w_u32le(h["start"]) + w_u32le(h["end"])
        shdr_data += w_u32le(h["start_loop"]) + w_u32le(h["end_loop"])
        shdr_data += w_u32le(h.get("rate", 44100))
        shdr_data += bytes([h.get("orig_pitch", 60) & 0xFF])
        shdr_data += struct.pack("<b", h.get("pitch_corr", 0))
        shdr_data += w_u16le(0)   # wSampleLink
        is_eos = h.get("end") == h.get("start")
        shdr_data += w_u16le(0 if is_eos else 1)  # sfSampleType: 0=terminal, 1=monoSample (avoids ROM prompt)
    if len(new_shdr) == 0 or (new_shdr[-1].get("end") != new_shdr[-1].get("start")):
        shdr_data += b"\x00" * 46
    pdta_body = (
        chunk("phdr", phdr_data) + chunk("pbag", pbag_data) + chunk("pmod", pmod_data) +
        chunk("pgen", pgen_data) + chunk("inst", inst_data) + chunk("ibag", ibag_data) +
        chunk("imod", imod_data) + chunk("igen", igen_data) + chunk("shdr", shdr_data)
    )
    pdta_list = b"LIST" + w_u32le(4 + len(pdta_body)) + b"pdta" + pdta_body
    if len(pdta_list) & 1:
        pdta_list += b"\x00"

    body = info_list + sdta_list + pdta_list
    riff = b"RIFF" + w_u32le(4 + len(body)) + b"sfbk" + body
    if len(riff) & 1:
        riff += b"\x00"
    with open(out_path, "wb") as f:
        f.write(riff)


# --- DLS: read with insh, filter, write ---

@dataclass
class DLSWaveParsed:
    fmt_data: bytes
    pcm_data: bytes
    wsmp_data: Optional[bytes]
    name: str


@dataclass
class DLSRegionParsed:
    rgnh: bytes  # 14 bytes payload
    wsmp: Optional[bytes]  # full chunk
    wlnk_payload: bytes   # 12 bytes: fusOptions(2), usPhaseGroup(2), ulChannel(4), ulTableIndex(4)
    art_data: Optional[bytes] = None


@dataclass
class DLSInstrParsed:
    ul_bank: int
    ul_instrument: int
    name: str
    regions: List[DLSRegionParsed]


def _parse_dls_raw(path: str) -> Tuple[List[DLSWaveParsed], List[DLSInstrParsed], bytes, Dict[str, bytes]]:
    data = open(path, "rb").read()
    root = _parse_riff(data)
    if root.list_type not in ("DLS ", "MLID"):
        raise RiffError("Not a DLS (RIFF DLS )")

    wvpl = _find_list(root, "wvpl")
    lins = _find_list(root, "lins")
    if not wvpl or not lins:
        raise RiffError("Malformed DLS: missing wvpl/lins")

    waves: List[DLSWaveParsed] = []
    for c in (wvpl.children or []):
        if c.fourcc != "LIST" or c.list_type != "wave":
            continue
        fmt = _find_child(c, "fmt ")
        dat = _find_child(c, "data")
        wsmp = _find_child(c, "wsmp")
        if not fmt or not dat:
            continue
        name = "wave"
        for info in (c.children or []):
            if info.fourcc == "LIST" and info.list_type == "INFO":
                inam = _find_child(info, "INAM")
                if inam and len(inam.data) > 8:
                    name = _read_zstr(inam.data[8:].tobytes())
        waves.append(DLSWaveParsed(
            fmt_data=fmt.data.tobytes(),
            pcm_data=dat.data.tobytes(),
            wsmp_data=wsmp.data.tobytes() if wsmp else None,
            name=name,
        ))

    instruments: List[DLSInstrParsed] = []
    for c in (lins.children or []):
        if c.fourcc != "LIST" or c.list_type != "ins ":
            continue
        insh = _find_child(c, "insh")
        lrgn = _find_list(c, "lrgn")
        if not insh or len(insh.data) < 12:
            continue
        c_regions = _read_u32le(insh.data, 0)
        ul_bank = _read_u32le(insh.data, 4)
        ul_instrument = _read_u32le(insh.data, 8)
        name = "Instrument"
        for info in (c.children or []):
            if info.fourcc == "LIST" and info.list_type == "INFO":
                inam = _find_child(info, "INAM")
                if inam and len(inam.data) > 8:
                    name = _read_zstr(inam.data[8:].tobytes())
        regions: List[DLSRegionParsed] = []
        for rgn_list in (lrgn.children or []):
            if rgn_list.fourcc != "LIST" or rgn_list.list_type not in ("rgn ", "rgn2"):
                continue
            rgnh = _find_child(rgn_list, "rgnh")
            wlnk = _find_child(rgn_list, "wlnk")
            wsmp = _find_child(rgn_list, "wsmp")
            if not rgnh or not wlnk or len(wlnk.data) < 12:
                continue
            table_index = _read_u32le(wlnk.data, 8)
            wlnk_payload = wlnk.data.tobytes() if len(wlnk.data) >= 12 else (b"\x00" * 12)
            if len(wlnk_payload) > 12:
                wlnk_payload = wlnk_payload[:12]
            elif len(wlnk_payload) < 12:
                wlnk_payload = wlnk_payload + b"\x00" * (12 - len(wlnk_payload))
            art_list = _find_list(rgn_list, "lar2")
            art_data = None
            if art_list:
                art_data = art_list.data.tobytes()
            regions.append(DLSRegionParsed(
                rgnh=rgnh.data.tobytes(),
                wsmp=wsmp.data.tobytes() if wsmp else None,
                wlnk_payload=wlnk_payload,
                art_data=art_data,
            ))
        instruments.append(DLSInstrParsed(
            ul_bank=ul_bank,
            ul_instrument=ul_instrument,
            name=name,
            regions=regions,
        ))

    info_chunks: Dict[str, bytes] = {}
    info_list = _find_list(root, "INFO")
    if info_list and info_list.children:
        for c in info_list.children:
            if c.fourcc == "INAM" and len(c.data) >= 8:
                info_chunks["INAM"] = c.data.tobytes()

    colh = _find_child(root, "colh")
    ptbl = _find_child(root, "ptbl")
    colh_data = colh.data.tobytes() if colh else b""
    ptbl_data = ptbl.data.tobytes() if ptbl else b""
    return waves, instruments, colh_data, info_chunks


def _dls_normalize_bank(bank: int, bank_style: str) -> int:
    if bank_style == "gm":
        return (bank >> 8) & 0x7F
    return bank & 0x7F7F


def _write_dls(
    out_path: str,
    waves: List[DLSWaveParsed],
    instruments: List[DLSInstrParsed],
    info_name: str = "Trimmed DLS",
) -> None:
    def w_u16(v: int) -> bytes:
        return struct.pack("<H", v & 0xFFFF)

    def w_u32(v: int) -> bytes:
        return struct.pack("<I", v & 0xFFFFFFFF)

    def w_i32(v: int) -> bytes:
        return struct.pack("<i", v)

    def list_chunk(list_type: str, body: bytes) -> bytes:
        size = 4 + len(body)
        out = b"LIST" + w_u32(size) + list_type.encode("ascii")[:4] + body
        if size & 1:
            out += b"\x00"  # RIFF chunk data pad to 2-byte boundary
        return out

    def pad_if_odd(data_len: int) -> bytes:
        return b"\x00" if (data_len & 1) else b""

    buf = bytearray()
    buf += b"RIFF"
    buf += b"\x00\x00\x00\x00"  # RIFF size (filled later); must be here so "DLS " stays at 8-11
    buf += b"DLS "
    buf += b"colh"
    buf += w_u32(4)
    buf += w_u32(len(instruments))
    lins_body = bytearray()
    for instr in instruments:
        ins_body = bytearray()
        ins_body += b"insh"
        ins_body += w_u32(12)
        ins_body += w_u32(len(instr.regions))
        ins_body += w_u32(instr.ul_bank)
        ins_body += w_u32(instr.ul_instrument)
        lrgn_body = bytearray()
        for rgn in instr.regions:
            rgn_body = bytearray()
            rgn_body += b"rgnh"
            rgn_body += w_u32(14)
            rgn_body += rgn.rgnh[:14]
            if rgn.wsmp:
                rgn_body += b"wsmp" + w_u32(len(rgn.wsmp)) + rgn.wsmp
            rgn_body += b"wlnk"
            rgn_body += w_u32(12)
            rgn_body += rgn.wlnk_payload[:12]
            if rgn.art_data:
                rgn_body += rgn.art_data
            lrgn_body += list_chunk("rgn2", bytes(rgn_body))
        ins_body += list_chunk("lrgn", bytes(lrgn_body))
        inam = (instr.name or "Instrument").encode("ascii") + b"\x00"
        ins_body += list_chunk("INFO", b"INAM" + w_u32(len(inam)) + inam)
        lins_body += list_chunk("ins ", bytes(ins_body))
    buf += list_chunk("lins", bytes(lins_body))

    # Build wvpl body first so we can compute ptbl offsets (relative to start of wvpl list content)
    wvpl_body = bytearray()
    wave_offsets: List[int] = []
    for w in waves:
        wave_offsets.append(len(wvpl_body))
        wave_list = bytearray()
        wave_list += b"fmt "
        wave_list += w_u32(len(w.fmt_data))
        wave_list += w.fmt_data
        wave_list += b"data"
        wave_list += w_u32(len(w.pcm_data))
        wave_list += w.pcm_data
        if len(w.pcm_data) & 1:
            wave_list += b"\x00"
        if w.wsmp_data:
            wave_list += b"wsmp" + w_u32(len(w.wsmp_data)) + w.wsmp_data
        inam = (w.name or "wave").encode("ascii") + b"\x00"
        wave_list += list_chunk("INFO", b"INAM" + w_u32(len(inam)) + inam)
        wvpl_body += list_chunk("wave", bytes(wave_list))

    ptbl_body = w_u32(8) + w_u32(len(waves))
    for off in wave_offsets:
        ptbl_body += w_u32(off)
    buf += b"ptbl"
    buf += w_u32(len(ptbl_body))
    buf += ptbl_body
    buf += pad_if_odd(len(ptbl_body))

    buf += list_chunk("wvpl", bytes(wvpl_body))

    info_body = b"INAM" + w_u32(len(info_name) + 1) + info_name.encode("ascii") + b"\x00"
    buf += list_chunk("INFO", info_body)

    buf[4:8] = w_u32(len(buf) - 8)  # RIFF size = rest of file after this field
    with open(out_path, "wb") as f:
        f.write(bytes(buf))


def _dls_filter_and_write(
    path: str,
    out_path: str,
    used_bank_program: Set[Tuple[int, int]],
    bank_style: str,
    dry_run: bool,
) -> None:
    waves, instruments, colh_data, info_chunks = _parse_dls_raw(path)
    used_inst: List[DLSInstrParsed] = []
    used_wave_indices: Set[int] = set()
    for instr in instruments:
        nb = _dls_normalize_bank(instr.ul_bank, bank_style)
        if (nb, instr.ul_instrument & 0x7F) in used_bank_program:
            used_inst.append(instr)
            for r in instr.regions:
                ti = struct.unpack_from("<I", r.wlnk_payload, 8)[0] if len(r.wlnk_payload) >= 12 else 0
                if 0 <= ti < len(waves):
                    used_wave_indices.add(ti)

    if not used_inst:
        raise ValueError("No instruments in the DLS match the (bank, program) used in the MIDI file.")

    used_wave_list = sorted(used_wave_indices)
    old_to_new_wave: Dict[int, int] = {old: new for new, old in enumerate(used_wave_list)}
    new_waves = [waves[i] for i in used_wave_list]
    new_instruments: List[DLSInstrParsed] = []
    for instr in used_inst:
        new_regions = []
        for r in instr.regions:
            ti = struct.unpack_from("<I", r.wlnk_payload, 8)[0] if len(r.wlnk_payload) >= 12 else -1
            if ti not in old_to_new_wave:
                continue
            new_payload = bytearray(r.wlnk_payload[:12])
            struct.pack_into("<I", new_payload, 8, old_to_new_wave[ti])
            new_regions.append(DLSRegionParsed(
                rgnh=r.rgnh,
                wsmp=r.wsmp,
                wlnk_payload=bytes(new_payload),
                art_data=r.art_data,
            ))
        if new_regions:
            new_instruments.append(DLSInstrParsed(
                ul_bank=instr.ul_bank,
                ul_instrument=instr.ul_instrument,
                name=instr.name,
                regions=new_regions,
            ))
    if not new_instruments:
        raise ValueError("No instruments left after filtering.")

    if dry_run:
        print(f"Would keep {len(new_instruments)} instruments, {len(new_waves)} waves.")
        return
    _write_dls(out_path, new_waves, new_instruments, info_name=info_chunks.get("INAM", b"")[:20].decode("ascii", "replace") or "Trimmed DLS")


# --- CLI ---

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Trim SF2/DLS to instruments used by a MIDI file (Program Change + Bank Select)."
    )
    ap.add_argument("midi", help="Input MIDI file")
    ap.add_argument("soundfont", help="Input SF2 or DLS file")
    ap.add_argument("-o", "--output", required=True, help="Output SF2 or DLS file")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be kept, do not write")
    ap.add_argument("--bank-style", choices=("mma", "gm"), default="mma", help="Bank match: mma (14-bit) or gm (7-bit)")
    args = ap.parse_args()

    if mido is None:
        print("Error: mido is required. Install with: pip install mido", file=sys.stderr)
        return 1

    try:
        used = collect_midi_bank_program(args.midi, args.bank_style)
    except Exception as e:
        print(f"Error reading MIDI: {e}", file=sys.stderr)
        return 1
    if not used:
        print("Warning: No (bank, program) found in MIDI.", file=sys.stderr)

    path = args.soundfont
    data = open(path, "rb").read()
    if len(data) < 12:
        print("Error: Soundfont file too small.", file=sys.stderr)
        return 1
    form = data[8:12].decode("ascii", "replace")
    if form == "sfbk":
        try:
            sf2 = _parse_sf2_raw(path)
            new_phdr, new_pbag, new_pgen, new_inst, new_ibag, new_igen, new_shdr, new_smpl, _, _ = _sf2_filter_and_remap(
                sf2, used, args.bank_style
            )
            if args.dry_run:
                print(f"Would keep {len(new_phdr)-1} presets, {len(new_inst)-1} instruments, {len(new_shdr)} samples.")
                return 0
            _write_sf2(args.output, sf2, new_phdr, new_pbag, new_pgen, new_inst, new_ibag, new_igen, new_shdr, new_smpl)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except RiffError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif form in ("DLS ", "MLID"):
        try:
            _dls_filter_and_write(path, args.output, used, args.bank_style, args.dry_run)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except RiffError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        print("Error: Not an SF2 or DLS file (RIFF form type must be sfbk or DLS ).", file=sys.stderr)
        return 1

    if not args.dry_run:
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
