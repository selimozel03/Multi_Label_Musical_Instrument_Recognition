import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from scipy.signal import resample_poly


# -------------------------
# OpenMIC 20-D tag order
# -------------------------
OPENMIC_TAGS = [
    "accordion","banjo","bass","cello","clarinet","cymbals","drums","flute","guitar",
    "mallet_percussion","mandolin","organ","piano","saxophone","synthesizer",
    "trombone","trumpet","ukulele","violin","voice"
]
tag2i = {t: i for i, t in enumerate(OPENMIC_TAGS)}


# -------------------------
# NSynth families used for audio generation
# -------------------------
NSYNTH_FAMILIES = [
    "bass","brass","flute","guitar","keyboard",
    "mallet","organ","reed","string","vocal","synth_lead"
]

# Families that produce supervised OpenMIC tags 
SUPERVISED_FAMILIES = ["bass","flute","guitar","keyboard","mallet","organ","vocal","synth_lead"]
UNSUPERVISED_ONLY_FAMILIES = ["brass","reed","string"]  # allowed in audio, but not supervised in OpenMIC


# -------------------------
# supervision policy (Path A)
# -------------------------
FAM_TO_OPENMIC = {
    "bass": ["bass"],
    "flute": ["flute"],
    "guitar": ["guitar"],
    "organ": ["organ"],
    "mallet": ["mallet_percussion"],
    "vocal": ["voice"],
    "keyboard": ["piano"],          # mask=1 as 
    "synth_lead": ["synthesizer"],  # mask=1 
}

UNSUPERVISED_TAGS = {
    # not generated at all
    "accordion","banjo","mandolin","ukulele","drums","cymbals",
    # ambiguous instrument tags (because we only have brass/reed/string families)
    "trumpet","trombone","clarinet","saxophone","violin","cello"
}
SUPERVISED_TAGS = set(OPENMIC_TAGS) - UNSUPERVISED_TAGS
# Expected: bass, flute, guitar, mallet_percussion, organ, piano, synthesizer, voice


# -------------------------
# Audio utilities
# -------------------------
def load_wav(path: Path):
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr

def resample(audio: np.ndarray, sr: int, target_sr: int):
    if sr == target_sr:
        return audio
    g = np.gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

def loop_to_len(audio: np.ndarray, target_len: int):
    if len(audio) >= target_len:
        return audio[:target_len]
    if len(audio) == 0:
        return np.zeros(target_len, dtype=np.float32)
    reps = int(np.ceil(target_len / len(audio)))
    return np.tile(audio, reps)[:target_len].astype(np.float32)

def peak_normalize(audio: np.ndarray, eps: float = 1e-8):
    m = np.max(np.abs(audio))
    return audio if m < eps else (audio / m).astype(np.float32)


# -------------------------
# Family selection logic
# -------------------------
def choose_families(k: int) -> list[str]:
    """
    Constraint:
    - k=0: []
    - k>=1: must include at least 1 supervised family
      (so k=1 can never be brass/reed/string only).
    """
    if k <= 0:
        return []

    # pick one supervised family first
    first = random.choice(SUPERVISED_FAMILIES)
    if k == 1:
        return [first]

    # then pick remaining from the rest of families (including brass/reed/string if desired)
    remaining_pool = [f for f in NSYNTH_FAMILIES if f != first]
    rest = random.sample(remaining_pool, k - 1)
    chosen = [first] + rest
    random.shuffle(chosen)
    return chosen


# -------------------------
# Main
# -------------------------
def main():
    random.seed(42)
    np.random.seed(42)

    nsynth_root = Path.home() / "Desktop" / "nsynth-train"
    examples_path = nsynth_root / "examples.json"
    audio_dir = nsynth_root / "audio"

    if not examples_path.exists():
        raise FileNotFoundError(f"examples.json not found at: {examples_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio/ not found at: {audio_dir}")

    out_root = Path.home() / "Desktop" / "synthetic_openmic20_from_nsynth_pathA"
    out_audio = out_root / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)

    # Audio config
    TARGET_SR = 16000
    DURATION_SEC = 10
    TARGET_LEN = TARGET_SR * DURATION_SEC

    # Schedule config
    N_PER_K = 200
    K_MIN = 1 
    K_MAX = len(NSYNTH_FAMILIES)  # 11 (k=0..11)

    # Build in-memory pools (FAST; no copying)
    with open(examples_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    ALLOW_SOURCES = {0, 1, 2}
    pool = {fam: [] for fam in NSYNTH_FAMILIES}

    for note_id, meta in examples.items():
        if meta.get("instrument_source") not in ALLOW_SOURCES:
            continue
        fam = meta.get("instrument_family_str")
        if fam in pool:
            wav = audio_dir / f"{note_id}.wav"
            if wav.exists():
                pool[fam].append(wav)

    print("Pool sizes:")
    for fam in NSYNTH_FAMILIES:
        print(f"  {fam:10s} {len(pool[fam])}")
        if len(pool[fam]) == 0:
            raise RuntimeError(f"Empty pool for family: {fam}")

    def make_y_m(chosen_families):
        # 20-D label vector
        y = np.zeros(20, dtype=np.int64)
        for fam in chosen_families:
            if fam in FAM_TO_OPENMIC:
                for tag in FAM_TO_OPENMIC[fam]:
                    y[tag2i[tag]] = 1

        # 20-D mask vector: supervise only SUPERVISED_TAGS
        m = np.zeros(20, dtype=np.int64)
        for tag in SUPERVISED_TAGS:
            m[tag2i[tag]] = 1
        return y, m

    rows = []
    mix_id = 0

    expected_rows = (K_MAX - K_MIN + 1) * N_PER_K
    print(f"\nGenerating mixtures: k={K_MIN}..{K_MAX}, {N_PER_K} each => {expected_rows} total")
    print("Constraint: every k>=1 includes >=1 supervised family (so k=1 is never brass/reed/string).")

    for k in range(K_MIN, K_MAX + 1):
        for _ in tqdm(range(N_PER_K), desc=f"polyphony={k}"):
            chosen = choose_families(k)

            waves = []
            for fam in chosen:
                wav_path = random.choice(pool[fam])
                w, sr = load_wav(wav_path)
                w = resample(w, sr, TARGET_SR)
                w = loop_to_len(w, TARGET_LEN)
                w = peak_normalize(w)
                gain = 10 ** (random.uniform(-10, 0) / 20.0)
                waves.append(gain * w)

            if len(waves) == 0:
                mix = np.zeros(TARGET_LEN, dtype=np.float32)
            else:
                mix = np.sum(np.stack(waves, axis=0), axis=0).astype(np.float32)
            mix = peak_normalize(mix)

            fname = f"mix_{mix_id:05d}.wav"
            sf.write(str(out_audio / fname), mix, TARGET_SR, subtype="PCM_16")

            y, m = make_y_m(chosen)

            row = {
                "filename": fname,
                "polyphony": k,
                "chosen_families": ",".join(chosen),
            }
            for t in OPENMIC_TAGS:
                row[f"y_{t}"] = int(y[tag2i[t]])
                row[f"m_{t}"] = int(m[tag2i[t]])
            rows.append(row)
            mix_id += 1

    df = pd.DataFrame(rows)
    labels_path = out_root / "labels.csv"
    df.to_csv(labels_path, index=False)

    print("\nDone.")
    print("Output root:", out_root)
    print("Audio dir:", out_audio)
    print("labels.csv:", labels_path)
    print("Rows:", len(df), "| Expected:", expected_rows)

    # Sanity check: ensure no k=1 rows with brass/reed/string
    k1 = df[df["polyphony"] == 1]["chosen_families"].value_counts()
    print("\npolyphony=1 family counts:")
    print(k1)

    print("\nSynthetic supervised tags (mask=1):", sorted(SUPERVISED_TAGS))
    print("Synthetic unsupervised tags (mask=0):", sorted(UNSUPERVISED_TAGS))


if __name__ == "__main__":
    main()
