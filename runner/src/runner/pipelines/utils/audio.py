"""This module provides functionality for converting audio files between different
formats.
"""

from io import BytesIO

import av
import numpy as np

class AudioConversionError(Exception):
    """Raised when an audio file cannot be converted."""

    def __init__(self, message="Audio conversion failed"):
        self.message = message
        super().__init__(self.message)


class AudioConverter:
    """Converts audio files to different formats."""

    @staticmethod
    def to_ndarray(input_bytes: bytes) -> np.ndarray:
        #inspired by https://github.com/SYSTRAN/faster-whisper/blob/d889345e071de21a83bdae60ba4b07110cfd0696/faster_whisper/audio.py
        """Converts audio in media file to a NumPy array.
        
        Args:
            input_bytes: The audio file as bytes to convert.

        Returns:
            The audio file as a NumPy array.
        """
        output_buffer = BytesIO()
        input_buffer = BytesIO(input_bytes)
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )
        
        audio_array = None
        try:
            input_container = av.open(input_buffer, mode="r")
            for stream in input_container.streams.audio:
                for frame in input_container.decode(stream):
                    resampled_frame = resampler.resample(frame)
                    array = resampled_frame[0].to_ndarray()
                    dtype = array.dtype
                    output_buffer.write(array)
            
            audio_array = np.frombuffer(output_buffer.getbuffer(), dtype=dtype)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
        except Exception as e:
            raise AudioConversionError(f"Error during audio conversion to numpy array: {e}")
        finally:
            input_container.close()

        return audio_array
    
    @staticmethod
    def convert(input_bytes: bytes, output_extension: str, output_codec: str) -> bytes:
        """Converts an audio file to a different format.

        Args:
            input_bytes: The audio file as bytes to convert.
            output_extension: The desired output format.
            output_codec: The desired output codec.

        Returns:
            The converted audio file as bytes.
        """
        if output_extension.startswith("."):
            output_extension = output_extension.lstrip(".")

        output_buffer = BytesIO()

        input_buffer = BytesIO(input_bytes)
        input_container = av.open(input_buffer)
        output_container = av.open(output_buffer, mode="w", format=output_extension)

        try:
            for stream in input_container.streams.audio:
                audio_stream = output_container.add_stream(
                    output_codec if output_codec else output_extension
                )

                # Convert input audio to target format.
                for frame in input_container.decode(stream):
                    for packet in audio_stream.encode(frame):
                        output_container.mux(packet)

                # Flush remaining packets to the output.
                for packet in audio_stream.encode():
                    output_container.mux(packet)
        except Exception as e:
            raise AudioConversionError(f"Error during audio conversion: {e}")
        finally:
            input_container.close()
            output_container.close()

        # Return the converted audio bytes.
        output_buffer.seek(0)
        converted_bytes = output_buffer.read()
        return converted_bytes
