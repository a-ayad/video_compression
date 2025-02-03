# video_compression



📽️ Included Video Encoders
🔹 H.264 / AVC
libx264 - Software-based H.264 encoder (best quality).
h264_amf - AMD AMF H.264 hardware encoder.
h264_nvenc - NVIDIA NVENC H.264 hardware encoder.
h264_qsv - Intel Quick Sync Video H.264 encoder.
🔹 H.265 / HEVC
libx265 - High-efficiency H.265 encoder (slow but best compression).
hevc_amf - AMD AMF H.265 encoder.
hevc_d3d12va - Direct3D 12 VA HEVC encoder.
hevc_nvenc - NVIDIA NVENC H.265 encoder.
hevc_qsv - Intel Quick Sync HEVC encoder.
🔹 AV1 (Next-Gen High-Efficiency)
libaom-av1 - Software-based AV1 encoder (high quality but slow).
librav1e - Rust-based AV1 encoder (balance of speed & quality).
libsvtav1 - Intel SVT-AV1 encoder (much faster than libaom).
av1_nvenc - NVIDIA NVENC AV1 encoder (RTX 40-series required).
av1_qsv - Intel Quick Sync Video AV1 encoder.
av1_amf - AMD AMF AV1 encoder.
🔹 VP8 & VP9 (Web Codecs)
libvpx - Open-source VP8 encoder.
vp8_vaapi - VP8 hardware-accelerated encoder (VAAPI).
libvpx-vp9 - Open-source VP9 encoder.
vp9_vaapi - VP9 hardware-accelerated encoder (VAAPI).
vp9_qsv - Intel Quick Sync VP9 encoder.

✅ Notes
NVENC, QSV, and VAAPI indicate hardware-accelerated encoding.
libx264, libx265, and libaom-av1 are software-based encoders with higher quality but slower speed.
