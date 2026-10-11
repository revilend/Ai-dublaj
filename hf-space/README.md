---
title: VoxStudio
emoji: 🎙️
colorFrom: violet
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
short_description: Ovoz klonlash va video dublyaj — XTTS-v2, whisper, NLLB
---

# 🎙️ VoxStudio — ovoz klonlash va video dublyaj

Bu Space — shu repozitoriydagi FastAPI studiyasining **telefon uchun** yengil
nashri: bitta `app.py` + `requirements.txt`.

| Imkoniyat | Model |
| --- | --- |
| Ovoz klonlash / matnni ovozga aylantirish | **Coqui XTTS-v2** (17 til) |
| Nutqni matnga aylantirish | **faster-whisper** |
| Tarjima | **NLLB-200** (`facebook/nllb-200-distilled-600M`) |
| Media (ajratish, moslash, mux) | **ffmpeg** |

## Nima qila oladi

1. **Ovoz klonlash** — 3–15 sekundlik namuna (yuklash yoki telefon mikrofoni)
   beriladi va istalgan matn o'sha ovozda o'qiladi.
2. **Video dublyaj** — video → matn (vaqt belgilari bilan) → tarjima → har bir
   satr klonlangan ovozda → video qayta yig'iladi. Ovoz namunasi bo'sh
   qoldirilsa, videoning o'z ovozi klonlanadi. Subtitr `.srt` fayli ham beriladi.

## Cheklovlar (bepul `cpu-basic`: 2 vCPU / 16 GB RAM)

* **O'zbekcha ovoz yo'q.** XTTS-v2 faqat 17 tilni biladi (en, ru, tr, es, fr,
  de, it, pt, ar, zh-cn, ja, ko, hi, nl, pl, cs, hu). O'zbek tili **manba**
  sifatida ishlaydi: o'zbekcha videoni boshqa tilga o'girish mumkin, aksi emas.
* Hisoblash sekin: taxminan har 1 daqiqa video uchun 2–5 daqiqa ishlov.
  Standart chegara — 10 daqiqalik video (`VOX_MAX_VIDEO_SECONDS`).
* **Musiqali fon saqlanmaydi** (Demucs bepul CPU'da juda sekin) — dublyaj ovozi
  asl trekni almashtiradi.
* Disk doimiy emas: Space qayta ishga tushsa, modellar (XTTS ≈ 1.9 GB,
  NLLB ≈ 2.4 GB) yana yuklanadi.

## Litsenziya

XTTS-v2 **Coqui Public Model License (CPML)** ostida — tijoriy bo'lmagan
foydalanish uchun. Ilova `COQUI_TOS_AGREED=1` ni o'zi o'rnatadi; bu shartni
bildiradi.
