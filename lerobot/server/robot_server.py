from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import subprocess
import os
import signal
import cv2
from typing import Dict, List, Optional
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
import glob
from pathlib import Path

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "static")), name="static")

# Global variables
teleoperation_process = None
output_queue = asyncio.Queue()
should_stream = False

# Add video capture devices dictionary
video_captures: Dict[int, cv2.VideoCapture] = {}

# Add after other constants
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/chrisheninger")


def get_video_capture(device_id: int) -> cv2.VideoCapture:
    if device_id not in video_captures:
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_captures[device_id] = cap
    return video_captures[device_id]


def generate_frames(device_id: int):
    cap = get_video_capture(device_id)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def cleanup_teleoperation():
    global teleoperation_process, should_stream
    teleoperation_process = None
    should_stream = False


async def process_output_reader(process):
    global should_stream
    try:
        while should_stream:
            line = await asyncio.get_event_loop().run_in_executor(
                None, process.stdout.readline
            )
            if not line:
                if process == teleoperation_process and process.poll() is not None:
                    print(
                        f"Teleoperation process exited with code: {process.returncode}")
                    cleanup_teleoperation()
                break
            await output_queue.put(line.strip())
    except Exception as e:
        print(f"Error in output reader: {str(e)}")
        cleanup_teleoperation()
    finally:
        if process.poll() is not None and process.returncode != 0:
            cleanup_teleoperation()
            await output_queue.put(f"Teleoperation failed with exit code {process.returncode}")


@app.get('/stream')
async def stream_output():
    async def event_generator():
        while should_stream:  # Only stream while should_stream is True
            if not output_queue.empty():
                yield {'data': await output_queue.get()}
            await asyncio.sleep(0.1)
        # Send a final message to indicate stream end
        yield {'data': '--- Streaming ended ---'}

    return EventSourceResponse(event_generator())


@app.post("/teleoperation/start")
async def start_teleoperation():
    global teleoperation_process, should_stream

    if teleoperation_process is not None:
        raise HTTPException(
            status_code=400, detail="Teleoperation is already running")

    try:
        script_path = os.path.join(os.path.dirname(
            __file__), "..", "scripts", "control_robot.py")
        robot_config = os.path.join(os.path.dirname(
            __file__), "..", "configs", "robot", "so100.yaml")

        # Simplified command with just essential parameters
        teleoperation_process = subprocess.Popen([
            "python", script_path, "teleoperate",
            "--robot-path", robot_config,
            "--robot-overrides", "~cameras",
            "--fps", "30",  # Add fixed FPS for stability
            "--display-cameras", "0"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

        should_stream = True
        asyncio.create_task(process_output_reader(teleoperation_process))

        # Give the process a moment to start
        await asyncio.sleep(1)

        return {"status": "started", "message": "Teleoperation started successfully"}
    except Exception as e:
        should_stream = False
        if teleoperation_process:
            try:
                teleoperation_process.terminate()
            except:
                pass
            teleoperation_process = None
        raise HTTPException(
            status_code=500, detail=f"Failed to start teleoperation: {str(e)}")


@app.post("/teleoperation/stop")
async def stop_teleoperation():
    global teleoperation_process, should_stream

    if teleoperation_process is None:
        raise HTTPException(
            status_code=400, detail="Teleoperation is not running")

    try:
        # Stop streaming
        should_stream = False

        teleoperation_process.send_signal(signal.SIGTERM)
        teleoperation_process.wait()
        teleoperation_process = None
        return {"status": "stopped", "message": "Teleoperation stopped successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to stop teleoperation: {str(e)}")


@app.get("/teleoperation/status")
async def get_teleoperation_status():
    return {
        "running": teleoperation_process is not None
    }


@app.get("/video/{device_id}")
async def video_feed(device_id: int):
    return StreamingResponse(
        generate_frames(device_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


def generate_unique_id():
    """Generate timestamp-based unique ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@app.post("/recording/start")
async def start_recording():
    global teleoperation_process, should_stream

    if teleoperation_process is not None:
        raise HTTPException(
            status_code=400, detail="Teleoperation is already running")

    try:
        unique_id = generate_unique_id()
        script_path = os.path.join(os.path.dirname(
            __file__), "..", "scripts", "control_robot.py")
        robot_config = os.path.join(os.path.dirname(
            __file__), "..", "configs", "robot", "so100.yaml")

        teleoperation_process = subprocess.Popen([
            "python", script_path, "record",
            "--robot-path", robot_config,
            "--fps", "30",
            "--repo-id", f"chrisheninger/so100_{unique_id}",
            "--warmup-time-s", "5",
            "--episode-time-s", "15",
            "--reset-time-s", "10",
            "--num-episodes", "1",
            "--push-to-hub", "0",
            "--single-task", "testing"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

        should_stream = True
        asyncio.create_task(process_output_reader(teleoperation_process))
        await asyncio.sleep(1)

        return {"status": "started", "recording_id": f"so100_{unique_id}"}
    except Exception as e:
        should_stream = False
        if teleoperation_process:
            try:
                teleoperation_process.terminate()
            except:
                pass
            teleoperation_process = None
        raise HTTPException(
            status_code=500, detail=f"Failed to start recording: {str(e)}")


@app.get("/recordings")
async def list_recordings():
    """Get list of available recordings"""
    recordings = []

    if os.path.exists(HF_CACHE_DIR):
        for dir_path in glob.glob(os.path.join(HF_CACHE_DIR, "so100_*")):
            if not os.path.isdir(dir_path):
                continue

            recording_id = os.path.basename(dir_path)
            # Only process if it starts with so100_
            if not recording_id.startswith("so100_"):
                continue

            clean_id = recording_id.replace("so100_", "", 1)
            recordings.append({
                "id": clean_id,
                "path": dir_path,
                "created": os.path.getctime(dir_path)
            })

    return sorted(recordings, key=lambda x: x["created"], reverse=True)


@app.post("/replay/{recording_id}")
async def replay_recording(recording_id: str):
    global teleoperation_process, should_stream

    if teleoperation_process is not None:
        raise HTTPException(
            status_code=400, detail="A process is already running")

    try:
        script_path = os.path.join(os.path.dirname(
            __file__), "..", "scripts", "control_robot.py")
        robot_config = os.path.join(os.path.dirname(
            __file__), "..", "configs", "robot", "so100.yaml")

        teleoperation_process = subprocess.Popen([
            "python", script_path, "replay",
            "--robot-path", robot_config,
            "--fps", "30",
            "--repo-id", f"chrisheninger/so100_{recording_id}",
            "--episode", "0"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

        should_stream = True
        asyncio.create_task(process_output_reader(teleoperation_process))
        await asyncio.sleep(1)

        return {"status": "started", "message": f"Replaying {recording_id}"}
    except Exception as e:
        should_stream = False
        if teleoperation_process:
            try:
                teleoperation_process.terminate()
            except:
                pass
            teleoperation_process = None
        raise HTTPException(
            status_code=500, detail=f"Failed to start replay: {str(e)}")


# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    for cap in video_captures.values():
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
