from instrumentalvocal import Session
import torch
import numpy as np
import numpy
from typing import List, Dict, Optional, Union



class InstrumentalDetector:
    def __init__(self):
        # Initialize the model from instrumentalvocal library
        self.model = Session()

    def detect(self, audio_path: str) -> Dict[str, Optional[List[Dict[str, Union[float, bool]]]]]:
        """
        Detects vocalness in an audio file. Tag 3 seconds audio segments as instrumental or vocal based on the vocalness score.
        A segment is considered vocal if its vocalness score is above 0.5.
        
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            Dict: A dictionary containing:
                - segments: List of dictionaries with segment start, end, vocalness score, and whether it's vocal.
                - vocal_segments: List of vocal segments, with start, end, and average vocalness score.
                - vocalness_score: Average vocalness score of the full audio track.
                - all_instrumental: Boolean indicating if the audio is fully instrumental (all segments detected as instrumental).
                - is_instrumental: Boolean indicating if the audio is considered instrumental based on the average vocalness score (vocalness_score < 0.2).
        """

        try:
            # Detect vocalness scores on 3 seconds segments
            detection_result = self.model.analyze(audio_path)

            segments_info = []
            vocal_segments = []
            #vocalness_scores = []
            #is_fully_instrumental = True

            last_end = 0.0

            for segment in detection_result.segments:
                vocalness = segment.vocalness
                
                # If the vocalness is above 0.5, we consider the segment as vocal
                is_vocal = vocalness > 0.5

                if is_vocal:
                    is_fully_instrumental = False

                segments_info.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "vocalness": vocalness,
                        "is_vocal": is_vocal,
                    }
                )

                # Merge consecutive vocal segments
                if is_vocal:
                    if segment.start == last_end and last_end != 0.0:
                        last_segment = vocal_segments[-1]
                        last_segment["end"] = segment.end
                        last_segment["vocalness"] = (
                            last_segment["vocalness"] + vocalness
                        ) / 2
                        vocal_segments[-1] = last_segment

                    else:
                        vocal_segments.append(
                            {
                                "start": segment.start,
                                "end": segment.end,
                                "vocalness": vocalness,
                            }
                        )
                    last_end = segment.end

                vocalness_scores.append(vocalness)

            return {
                "segments": segments_info,
                "vocal_segments": vocal_segments,
                "vocalness_score": np.mean(vocalness_scores),
                #"all_instrumental": is_fully_instrumental,
                #"is_instrumental": np.mean(vocalness_scores) < 0.2
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {
                "segments": [],
                "vocal_segments": [],
                "vocalness_score": 0.0,
                "all_instrumental": False,
                "is_instrumental": False,
            }
