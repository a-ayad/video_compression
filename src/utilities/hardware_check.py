import subprocess

def get_available_encoders():
    """Get a list of available ffmpeg encoders."""
    result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
    encoders = result.stdout.split("\n")

    available_encoders = []
    for line in encoders:
        if line.strip() and not line.startswith("------"):
            parts = line.split()
            if len(parts) > 1:
                available_encoders.append(parts[1])  # Extract encoder name

    return available_encoders

def is_qsv_supported():
    """Check if Intel Quick Sync (QSV) is available and working."""
    encoders = get_available_encoders()
    if "h264_qsv" not in encoders:
        return False  # QSV encoder is not installed

    # Try running a test encoding command
    test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'nullsrc=s=1280x720:d=1', 
                '-c:v', 'h264_qsv', '-f', 'null', '-']
    
    result = subprocess.run(test_cmd, stderr=subprocess.PIPE, text=True)
    print(result.stderr)
    if "Conversion failed!" in result.stderr:
        return False # QSV is installed but not working
    # Check if there is an error indicating hardware failure


    return True  # QSV is working
print("Checking hardware acceleration support...")
print("Available encoders:", get_available_encoders())
if is_qsv_supported():
    print("✅ Intel Quick Sync Video (QSV) is fully supported and working!")
else:
    print("❌ Intel QSV is NOT available or not working properly.")
