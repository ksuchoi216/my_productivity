import argparse
import base64
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

# Load .env via python-dotenv
try:
    from dotenv import load_dotenv  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "python-dotenv is required to load .env. Install with: pip install python-dotenv"
    ) from exc

load_dotenv()


@dataclass
class Config:
    anki_url: str
    deck: str
    target_deck: str
    front_field: str
    back_field: str
    audio_dir: Path
    sentence_dir: Path
    tts_model: str
    tts_voice: str
    tts_instructions: str
    text_model: str
    dry_run: bool
    limit: int | None
    overwrite_audio: bool
    overwrite_sentence: bool


def load_config(args: argparse.Namespace) -> Config:
    deck = args.deck or os.getenv("ANKI_DECK", "hackers_toefl")
    target_deck = os.getenv("ANKI_TARGET_DECK", f"{deck}_new")
    return Config(
        anki_url=args.anki_url or os.getenv("ANKI_CONNECT_URL", "http://localhost:8760"),
        deck=deck,
        target_deck=target_deck,
        front_field=args.front_field or os.getenv("ANKI_FRONT_FIELD", "Front"),
        back_field=args.back_field or os.getenv("ANKI_BACK_FIELD", "Back"),
        audio_dir=Path(os.getenv("ANKI_AUDIO_DIR", "anki_handler/audio")),
        sentence_dir=Path(os.getenv("ANKI_SENTENCE_DIR", "anki_handler/sentence")),
        tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        tts_voice=os.getenv("OPENAI_TTS_VOICE", "coral"),
        tts_instructions=os.getenv(
            "OPENAI_TTS_INSTRUCTIONS",
            "Speak in a natural British English accent and pronounce the word clearly.",
        ),
        text_model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini"),
        dry_run=args.dry_run,
        limit=args.limit,
        overwrite_audio=args.overwrite_audio,
        overwrite_sentence=args.overwrite_sentence,
    )


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to .env or export it before running."
        )


def anki_request(anki_url: str, action: str, **params):
    resp = requests.post(
        anki_url, json={"action": action, "version": 6, "params": params}
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise RuntimeError(f"AnkiConnect error: {payload['error']}")
    return payload.get("result")


def ensure_deck(anki_url: str, deck_name: str) -> None:
    anki_request(anki_url, "createDeck", deck=deck_name)


def fetch_notes(anki_url: str, deck: str) -> Dict[int, Dict[str, str]]:
    card_ids = anki_request(anki_url, "findCards", query=f'deck:"{deck}"')
    if not card_ids:
        return {}
    cards = anki_request(anki_url, "cardsInfo", cards=card_ids)

    notes: Dict[int, Dict[str, str]] = {}
    for card in cards:
        note_id = card.get("note")
        fields = card.get("fields", {})
        field_values = {name: info.get("value", "") for name, info in fields.items()}
        if note_id not in notes:
            notes[note_id] = field_values
    return notes


def fetch_note_infos(anki_url: str, note_ids: List[int]) -> List[Dict]:
    if not note_ids:
        return []
    return anki_request(anki_url, "notesInfo", notes=note_ids) or []


def extract_fields(note_info: Dict) -> Dict[str, str]:
    fields = note_info.get("fields", {})
    out: Dict[str, str] = {}
    for name, info in fields.items():
        if isinstance(info, dict):
            out[name] = info.get("value", "")
        else:
            out[name] = str(info)
    return out


def get_or_create_note_copy(
    anki_url: str, note_info: Dict, target_deck: str, source_note_id: int
) -> int:
    tag = f"src_note_{source_note_id}"
    existing = anki_request(
        anki_url, "findNotes", query=f'deck:"{target_deck}" tag:"{tag}"'
    )
    if existing:
        return existing[0]

    fields = extract_fields(note_info)
    tags = list(note_info.get("tags", []))
    if tag not in tags:
        tags.append(tag)

    note = {
        "deckName": target_deck,
        "modelName": note_info.get("modelName"),
        "fields": fields,
        "tags": tags,
        "options": {"allowDuplicate": True},
    }
    new_id = anki_request(anki_url, "addNote", note=note)
    if not new_id:
        raise RuntimeError(f"Failed to add note copy for source note {source_note_id}")
    return int(new_id)


def prepare_target_notes(
    anki_url: str,
    source_notes: Dict[int, Dict[str, str]],
    target_deck: str,
    limit: int | None,
) -> Dict[int, Dict[str, str]]:
    ensure_deck(anki_url, target_deck)

    source_note_ids: List[int] = []
    for note_id, _ in iter_notes(source_notes, limit):
        source_note_ids.append(note_id)

    if not source_note_ids:
        return {}

    source_infos = fetch_note_infos(anki_url, source_note_ids)
    info_map = {info.get("noteId"): info for info in source_infos if info}

    # Map existing target notes by src_note_<id> tag to avoid duplicates
    existing_target_note_ids = anki_request(
        anki_url, "findNotes", query=f'deck:"{target_deck}"'
    ) or []
    existing_map: Dict[int, int] = {}
    if existing_target_note_ids:
        existing_infos = fetch_note_infos(anki_url, existing_target_note_ids)
        for info in existing_infos:
            if not info:
                continue
            tags = info.get("tags", [])
            if not tags:
                continue
            for tag in tags:
                if tag.startswith("src_note_"):
                    try:
                        src_id = int(tag.replace("src_note_", ""))
                        existing_map[src_id] = info.get("noteId")
                    except ValueError:
                        continue

    target_note_ids: List[int] = []
    for source_note_id in source_note_ids:
        note_info = info_map.get(source_note_id)
        if not note_info:
            continue
        if source_note_id in existing_map:
            target_note_ids.append(existing_map[source_note_id])
            continue
        target_id = get_or_create_note_copy(
            anki_url, note_info, target_deck, source_note_id
        )
        target_note_ids.append(target_id)

    target_infos = fetch_note_infos(anki_url, target_note_ids)
    return {info.get("noteId"): extract_fields(info) for info in target_infos if info}


def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text or "word"


def ensure_dirs(audio_dir: Path, sentence_dir: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    sentence_dir.mkdir(parents=True, exist_ok=True)


def build_audio_filename(front: str) -> str:
    safe = sanitize_filename(front)
    return f"anki_uk_{safe}.mp3"


def build_sentence_filename(front: str) -> str:
    safe = sanitize_filename(front)
    return f"anki_sentence_{safe}.txt"


def build_legacy_audio_filename(note_id: int, front: str) -> str:
    safe = sanitize_filename(front)
    return f"anki_uk_{note_id}_{safe}.mp3"


def build_legacy_sentence_filename(note_id: int, front: str) -> str:
    safe = sanitize_filename(front)
    return f"anki_sentence_{note_id}_{safe}.txt"


def resolve_asset_paths(
    note_id: int, front: str, audio_dir: Path, sentence_dir: Path
) -> Tuple[Path, Path]:
    audio_filename = build_audio_filename(front)
    sentence_filename = build_sentence_filename(front)

    audio_path = audio_dir / audio_filename
    sentence_path = sentence_dir / sentence_filename

    if not audio_path.exists():
        legacy_audio = audio_dir / build_legacy_audio_filename(note_id, front)
        if legacy_audio.exists():
            audio_path = legacy_audio

    if not sentence_path.exists():
        legacy_sentence = sentence_dir / build_legacy_sentence_filename(note_id, front)
        if legacy_sentence.exists():
            sentence_path = legacy_sentence

    return audio_path, sentence_path


def build_new_back(existing: str, audio_tag: str, example_sentence: str) -> str:
    existing = existing or ""
    additions: List[str] = []

    if audio_tag and audio_tag not in existing:
        additions.append(audio_tag)

    if example_sentence:
        example_text = example_sentence.replace("\n", "<br>")
        example_line = f"Example: {example_text}"
        if example_line not in existing:
            additions.append(example_line)

    if not additions:
        return existing

    spacer = "<br><br>" if existing.strip() else ""
    return existing + spacer + "<br>".join(additions)


def create_tts_audio_bytes(text: str, model: str, voice: str, instructions: str) -> bytes:
    from openai import OpenAI

    client = OpenAI()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="mp3",
        ) as response:
            response.stream_to_file(tmp_path)

        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def store_audio_in_anki(
    anki_url: str, filename: str, audio_bytes: bytes, overwrite: bool
) -> str:
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    if overwrite:
        # AnkiConnect storeMediaFile does not accept an overwrite flag.
        # Delete first if present; ignore errors if missing.
        try:
            anki_request(anki_url, "deleteMediaFile", filename=filename)
        except Exception:
            pass
    anki_request(
        anki_url,
        "storeMediaFile",
        filename=filename,
        data=b64,
    )
    return f"[sound:{filename}]"


def build_example_sentence(word: str, model: str, num_sentences: int = 3) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an English tutor. Produce exactly {num_sentences} short, natural English sentences using the given word. "
                "Use British spelling when possible. Return only the sentences, one per line, with no numbering, quotes, or extra text.",
            ),
            ("human", "Word: {word}"),
        ]
    )

    llm = ChatOpenAI(model=model, temperature=0.7)
    messages = prompt.format_messages(word=word, num_sentences=num_sentences)
    result = llm.invoke(messages)
    raw = (result.content or "").strip()
    if not raw:
        return ""

    lines = []
    for line in raw.splitlines():
        line = line.strip().strip("\"'")
        if line:
            lines.append(line)

    if len(lines) >= num_sentences:
        lines = lines[:num_sentences]

    return "\n".join(lines)


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def generate_assets(
    note_id: int,
    fields: Dict[str, str],
    front_field: str,
    audio_dir: Path,
    sentence_dir: Path,
    tts_model: str,
    tts_voice: str,
    tts_instructions: str,
    text_model: str,
    num_sentences: int,
    overwrite_audio: bool,
    overwrite_sentence: bool,
) -> Tuple[str, Path, Path]:
    front = fields.get(front_field, "").strip()
    if not front:
        raise ValueError("Front is empty")

    audio_filename = build_audio_filename(front)
    sentence_filename = build_sentence_filename(front)

    audio_path = audio_dir / audio_filename
    sentence_path = sentence_dir / sentence_filename

    sentence_text = ""
    if overwrite_sentence or not sentence_path.exists():
        sentence_text = build_example_sentence(
            front, text_model, num_sentences=num_sentences
        )
        sentence_path.write_text(sentence_text, encoding="utf-8")

    audio_needs = overwrite_audio or not audio_path.exists()
    if audio_needs:
        if not sentence_text:
            if sentence_path.exists():
                sentence_text = sentence_path.read_text(encoding="utf-8")
            else:
                sentence_text = build_example_sentence(
                    front, text_model, num_sentences=num_sentences
                )
                sentence_path.write_text(sentence_text, encoding="utf-8")

        first_sentence = first_non_empty_line(sentence_text)
        if first_sentence:
            tts_input = f"{front}, {first_sentence}"
        else:
            tts_input = front
        audio_bytes = create_tts_audio_bytes(
            tts_input, tts_model, tts_voice, tts_instructions
        )
        audio_path.write_bytes(audio_bytes)

    return front, audio_path, sentence_path


def load_assets(audio_path: Path, sentence_path: Path) -> Tuple[bytes, str]:
    audio_bytes = audio_path.read_bytes()
    sentence = sentence_path.read_text(encoding="utf-8").strip()
    return audio_bytes, sentence


def update_note_fields(
    anki_url: str, note_id: int, fields: Dict[str, str]
) -> None:
    anki_request(anki_url, "updateNoteFields", note={"id": note_id, "fields": fields})


def iter_notes(
    notes: Dict[int, Dict[str, str]], limit: int | None
) -> Iterable[Tuple[int, Dict[str, str]]]:
    count = 0
    for note_id, fields in notes.items():
        yield note_id, fields
        count += 1
        if limit is not None and count >= limit:
            break


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate UK pronunciation audio + example sentence and update Anki notes."
    )
    parser.add_argument("--deck", help="Anki deck name")
    parser.add_argument("--anki-url", help="AnkiConnect URL")
    parser.add_argument("--front-field", help="Front field name", default=None)
    parser.add_argument("--back-field", help="Back field name", default=None)
    parser.add_argument("--limit", type=int, help="Limit number of notes to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not update notes")
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Overwrite existing audio files with same filename",
    )
    parser.add_argument(
        "--overwrite-sentence",
        action="store_true",
        help="Overwrite existing sentence files with same filename",
    )

    args = parser.parse_args()
    if args.limit == 0:
        args.limit = None
    config = load_config(args)

    require_api_key()

    source_notes = fetch_notes(config.anki_url, config.deck)
    if not source_notes:
        print(f"No notes found in deck: {config.deck}")
        return 0

    target_notes = prepare_target_notes(
        config.anki_url, source_notes, config.target_deck, config.limit
    )
    if not target_notes:
        print(f"No notes prepared in target deck: {config.target_deck}")
        return 0

    ensure_dirs(config.audio_dir, config.sentence_dir)

    processed = 0
    skipped = 0

    # Phase 1: generate assets to local folders
    num_sentences = int(os.getenv("ANKI_NUM_SENTENCES", "3"))
    for note_id, fields in iter_notes(target_notes, config.limit):
        try:
            generate_assets(
                note_id,
                fields,
                config.front_field,
                config.audio_dir,
                config.sentence_dir,
                config.tts_model,
                config.tts_voice,
                config.tts_instructions,
                config.text_model,
                num_sentences,
                config.overwrite_audio,
                config.overwrite_sentence,
            )
        except Exception as exc:
            front = fields.get(config.front_field, "").strip()
            print(f"[WARN] generate failed note={note_id} word={front}: {exc}")
            skipped += 1

    # Phase 2: load assets and update notes
    for note_id, fields in iter_notes(target_notes, config.limit):
        front = fields.get(config.front_field, "").strip()
        back = fields.get(config.back_field, "")

        if not front:
            skipped += 1
            continue

        audio_path, sentence_path = resolve_asset_paths(
            note_id, front, config.audio_dir, config.sentence_dir
        )
        audio_filename = audio_path.name

        if not audio_path.exists() or not sentence_path.exists():
            print(f"[WARN] missing assets note={note_id} word={front}")
            skipped += 1
            continue

        try:
            audio_bytes, example_sentence = load_assets(audio_path, sentence_path)
            audio_tag = store_audio_in_anki(
                config.anki_url, audio_filename, audio_bytes, config.overwrite_audio
            )
            new_back = build_new_back(back, audio_tag, example_sentence)
        except Exception as exc:
            print(f"[WARN] update prep failed note={note_id} word={front}: {exc}")
            skipped += 1
            continue

        if config.dry_run:
            print(f"[DRY RUN] note={note_id} word={front} -> back+=audio+example")
        else:
            update_note_fields(
                config.anki_url,
                note_id,
                {
                    config.front_field: front,
                    config.back_field: new_back,
                },
            )

        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
