
# Discogs-VI: Audio Encoder Dataset

## 1. Discogs-VI matched with Deezer (metadata)

`data/audio_encoder_dataset/metadata/full_metadata.csv`
643,342
columns : 'version_id', 'clique_id', 'radar_id', 'deezer_track_id', 'md5_encoded', 'encoding'
description: discogs_vi_dataset matched with deezer collection

## 2. Discogs-VI - 30sec chunks

`data/audio_encoder_dataset/metadata/full.csv`
2,087,761
columns: 'version_id', 'clique_id', 'md5_encoded', 'deezer_id', 'chunk_id', 'start', 'end', 'duration', 'transcription'
description: data chunked (30s) with transcriptions + start/end of vocal segments

## 3. Discogs-VI - full transcription

`data/audio_encoder_dataset/metadata/transcriptions.csv`
402,254
columns:'file', 'transcription_vocal', 'md5_encoded'
description: transcriptions results from discogs_vi

## 4. Discogs-VI - vocal detection

`data/audio_encoder_dataset/metadata/vocal_detector_results.csv`
415,680
columns = 'file', 'is_instrumental', 'res_detection', 'vocalness_score', 'vocal_segments', 'per_instrumental_segments', 'md5_encoded'
description: results from vocal detector

## 5. Discogs-VI - 30sec chunks  (+ vocalness scores)

`data/audio_encoder_dataset/metadata/full2.csv`
2,086,110
columns =  'version_id', 'clique_id', 'md5_encoded', 'deezer_id', 'chunk_id', 'start', 'end', 'duration', 'vocalness', 'transcription'
description: data chunked (30s) with transcriptions + start/end of vocal segments (v2)

## 6. Discogs-VI - 30sec chunks + filtered to only keep 30s

`data/audio_encoder_dataset/metadata/full_filtered.csv`
1,689,432
columns = 'version_id', 'clique_id', 'md5_encoded', 'deezer_id', 'chunk_id',
       'start', 'end', 'duration', 'transcription'
description: full filtered to only keep 30s chunks

---

# Covers80

## Covers80 : final benchmark

`data/benchmarks/covers80.csv`
116
cols : 'md5_encoded', 'encoding', 'md5_origin', 'version_id', 'clique_id',
       'lyrics', 'title', 'artist', 'file', 'is_instrumental', 'res_detection',
       'vocalness_score', 'vocal_segments', 'per_instrumental_segments',
       'transcription_vocal', 'transcription_vocal_english', 'transcription'
desc: covers80

---

# SHS100k

## SHS100k : Initial subset matched to Deezer via Youtube and Radar

`data/benchmarks/shs100k/shs100k_init.csv`
1,441
'version_id', 'clique_id', 'title', 'artist', 'md5_origin', 'deezer_id',
       'lyrics_id', 'lyrics', 'encoding', 'md5_encoded'

`data/benchmarks/shs100k/shs100k_res_vocal_detection.csv`
1,441
'file', 'is_instrumental', 'res_detection', 'vocalness_score',
       'vocal_segments', 'per_instrumental_segments'
pass after vocal detection

## SHS100k : Output from vocal detection

`data/benchmarks/shs100k/shs100k_init.csv`
1,086
cols: 'md5_encoded', 'is_instrumental', 'all_instrumental', 'vocalness_score',
       'res_detection', 'vocal_segments', 'version_id', 'clique_id', 'title',
       'artist', 'md5_origin', 'deezer_id', 'lyrics_id', 'lyrics', 'encoding'

## SHS100k: Final benchmark

`data/benchmarks/shs100k/shs100k.csv`
1,076
cols: 'md5_encoded', 'transcription_vocal', 'is_instrumental',
       'all_instrumental', 'vocalness_score', 'res_detection',
       'vocal_segments', 'version_id', 'clique_id', 'title', 'artist',
       'md5_origin', 'deezer_id', 'lyrics_id', 'lyrics', 'encoding'

## SHS100k : Q1

`data/benchmarks/shs100k/shs100k_q1.csv`
714
cols : 'md5_encoded', 'is_instrumental', 'all_instrumental', 'vocalness_score',
       'res_detection', 'vocal_segments', 'version_id', 'clique_id', 'title',
       'artist', 'md5_origin', 'deezer_id', 'lyrics_id', 'lyrics', 'encoding',
       'nb_vocal_segments', 'nb_audio_chunks'

## SHS100k : Q2

`data/benchmarks/shs100k/shs100k_median.csv`
374
cols : 'md5_encoded', 'is_instrumental', 'all_instrumental', 'vocalness_score',
       'res_detection', 'vocal_segments', 'version_id', 'clique_id', 'title',
       'artist', 'md5_origin', 'deezer_id', 'lyrics_id', 'lyrics', 'encoding',
       'nb_vocal_segments', 'nb_audio_chunks'

## SHS100k : Q3

`data/benchmarks/shs100k/shs100k_q3.csv`
133
cols : 'md5_encoded', 'is_instrumental', 'all_instrumental', 'vocalness_score',
       'res_detection', 'vocal_segments', 'version_id', 'clique_id', 'title',
       'artist', 'md5_origin', 'deezer_id', 'lyrics_id', 'lyrics', 'encoding',
       'nb_vocal_segments', 'nb_audio_chunks'

---

# Discogs-VI

## Discogs-VI: Output from vocal detection

`data/benchmarks/discogs_vi/discogs_vi.csv`
83,570
cols: 'md5_encoded', 'encoding', 'id', 'clique_id', 'version_id',
       'md5_origin', 'deezer_track_id', 'is_instrumental', 'all_instrumental',
       'vocalness_score', 'res_detection', 'vocal_segments',
       'transcription_vocal'

## Discogs-VI: Q1

`data/benchmarks/discogs_vi/discogs_vi_init.csv`
121,337
cols: 'md5_encoded', 'encoding', 'id', 'clique_id', 'version_id', 'md5_origin', 'deezer_track_id', 'is_instrumental', 'all_instrumental', 'vocalness_score', 'res_detection', 'vocal_segments’

## Discogs-VI: Q2

`data/benchmarks/discogs_vi/discogs_vi_q1.csv`
56,538
cols: ['md5_encoded', 'encoding', 'id', 'clique_id', 'version_id',
       'md5_origin', 'deezer_track_id', 'is_instrumental', 'all_instrumental',
       'vocalness_score', 'res_detection', 'vocal_segments',
       'nb_vocal_segments', 'nb_audio_chunks']

## Discogs-VI: Q2

`data/benchmarks/discogs_vi/discogs_vi_median.csv`
27,799
cols: ['md5_encoded', 'encoding', 'id', 'clique_id', 'version_id',
       'md5_origin', 'deezer_track_id', 'is_instrumental', 'all_instrumental',
       'vocalness_score', 'res_detection', 'vocal_segments',
       'nb_vocal_segments', 'nb_audio_chunks']

## Discogs-VI: Q3

`data/benchmarks/discogs_vi/discogs_vi_q3.csv`
7,275
cols: ['md5_encoded', 'encoding', 'id', 'clique_id', 'version_id',
       'md5_origin', 'deezer_track_id', 'is_instrumental', 'all_instrumental',
       'vocalness_score', 'res_detection', 'vocal_segments',
       'nb_vocal_segments', 'nb_audio_chunks']

---

# SHS100k-mini

## SHS100k-mini: Final subset

`data/benchmarks/shs100k_mini/shs100k_mini.csv`
167
cols: ['md5_encoded', 'encoding', 'md5_origin', 'version_id', 'clique_id',
       'deezer_id', 'title', 'artist', 'lyrics', 'file', 'is_instrumental',
       'res_detection', 'vocalness_score', 'vocal_segments',
       'per_instrumental_segments', 'transcription_vocal', 'transcription',
       'transcription_vocal_english'],

---

# Discogs-VI-mini

## Discogs-VI-mini: Final benchmark

`data/benchmarks/discogs_vi_mini/discogs_vi_mini.csv`
4,283
cols: 'version_id', 'clique_id', 'md5_origin', 'deezer_id', 'lyrics',
       'md5_encoded', 'encoding', 'file', 'is_instrumental', 'res_detection',
       'vocalness_score', 'vocal_segments', 'per_instrumental_segments',
       'transcription_vocal', 'transcription_vocal_english', 'transcription'
