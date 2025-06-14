import asyncio
import pyaudio
import threading
import os
import queue
import json
import numpy as np
import time
from time import sleep
import logging
import cv2
import torch
import shutil
import concurrent.futures

# Robot control imports
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.robots.utils import make_robot_config, make_robot
from lerobot.common.robot_devices.utils import busy_wait

# Transcription imports
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from dotenv import load_dotenv

load_dotenv()
# OpenAI imports
from openai import OpenAI
from dotenv import load_dotenv

# Import for remote inference
from lerobot_client import LeRobotClient
from datetime import datetime

# Global robot instance for stop functionality
robot_controller = None
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class Configuration:
    # 16:9 aspect ratio
    LD = (426, 240)
    SD = (854, 480)
    HD = (1280, 720)
    FHD = (1920, 1080)

    QPOS_MAP = {
        "zero": [0, 0, 0, 0, 0, 0],
        "rotated": [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2],
        "rest": [0.049, -3.62, 3.19, 1.26, -0.17, -0.67]
    }

    POS_MAP = {
        "zero": [2035, 3081, 1001, 1966, 1988, 2125],
        "rotated": [3052, 2021, 2040, 3062, 905, 3179],
        "rest": [2068, 819, 3051, 2830, 2026, 2049],
    }

    DOFS = 6
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

class Feetech():
    ROBOT_TYPE = 'so100'
    ARM_NAME = 'main'
    ARM_TYPE = 'follower'

    MODEL_RESOLUTION = 4096
    RADIAN_PER_STEP = (2 * np.pi) / MODEL_RESOLUTION
    MOTOR_DIRECTION = [-1, 1, 1, 1, 1, 1]
    JOINT_IDS = [0, 1, 2, 3, 4, 5]
    ## replace with your own port
    PORT = '/dev/tty.usbmodem59700741371'
    REFERENCE_FRAME = 'rotated'

    def move_to_pos(self, pos):
        self.move(pos)

    def __init__(self, **kwargs):
        self.qpos_handler = kwargs.get('qpos_handler', None)
        connect = kwargs.get('connect', True)
        if connect:
            self.connect()

    def connect(self):
        self.motors_bus = self._create_motors_bus()

    def disconnect(self):
        self.motors_bus.disconnect()

    def get_pos(self):
        return self.motors_bus.read('Present_Position')

    def move(self, target_pos):
        self.control_position(target_pos)
        position = self.get_pos()
        print(f"Current position: {position}")
        print(f"Target position: {target_pos}")
        error = np.linalg.norm(target_pos - position) / Feetech.MODEL_RESOLUTION
        print(f"Position error: {error}")

    def go_to_rest(self):
        self.go_to_preset('rest')

    def go_to_preset(self, preset):
        pos = Configuration.POS_MAP[preset]
        self.move(pos)
        time.sleep(1)
        self.disconnect()

    def _create_motors_bus(self):
        robot_config = make_robot_config(Feetech.ROBOT_TYPE)
        motors = robot_config.follower_arms[Feetech.ARM_NAME].motors
        config = FeetechMotorsBusConfig(port=self.PORT, motors=motors)
        motors_bus = FeetechMotorsBus(config)
        motors_bus.connect()
        return motors_bus

    def _motor_names(self, ids):
        return [
            self._motor_name(id)
            for id in ids
        ]

    def _motor_name(self, id):
        return Configuration.JOINT_NAMES[id]
    
    def control_position(self, pos):
        self.motors_bus.write('Goal_Position', pos)


# Robot control function
def go_to_position(position):
    """
    Move to a specific position
    
    Args:
        position (str): Target position - can be 'rest', 'zero', or 'rotated'
    
    Returns:
        dict: Status of the movement
    """
    valid_positions = ['rest', 'zero', 'rotated']
    
    if position not in valid_positions:
        return {
            "status": "error",
            "message": f"Invalid position '{position}'. Valid positions are: {', '.join(valid_positions)}"
        }
    
    try:
        print(f"🤖 Moving to position: {position}")
        feetech = Feetech()
        feetech.move(Configuration.POS_MAP[position])
        feetech.disconnect()
        
        return {
            "status": "success",
            "message": f"Successfully moved to {position} position",
            "current_position": position
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to move to {position}: {str(e)}"
        }


async def grasp_and_drop(object_to_grasp, drop_location, inference_time_s=30, fps=25, device="mps", 
                        robot_type="so100", output_dir="images/", websocket_url="ws://localhost:8765"):
    """
    Perform grasp and drop task using remote inference
    
    Args:
        object_to_grasp (str): Object to grasp (e.g., "yellow cuboid", "red ball")
        drop_location (str): Where to drop the object (e.g., "bin", "table", "blue container")
        inference_time_s (int): Inference time in seconds (default: 30)
        fps (int): Frames per second (default: 25)
        device (str): Device to use (default: "mps")
        robot_type (str): Robot type (default: "so100")
        output_dir (str): Output directory for images (default: "images/")
        websocket_url (str): WebSocket server URL (default: "ws://localhost:8765")
    
    Returns:
        dict: Status of the grasp and drop operation
    """
    # Create task description
    task = f"Grasp the {object_to_grasp} and put it in the {drop_location}."
    robot = None
    client = None
    
    try:
        print(f"🤖 Starting grasp and drop task: {task}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)

        # Initialize robot (policy is handled by remote WebSocket server)
        robot = make_robot(robot_type)
        robot.connect()
        print("✅ Robot connected")

        # Setup output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Performance tracking variables
        iteration_times = []
        running_total_time = 0.0
        successful_steps = 0
        start_overall = time.perf_counter()

        # Use async context manager for LeRobotClient
        async with LeRobotClient(websocket_url) as client:
            print("✅ LeRobot client connected and ready")
            
            try:
                # Main inference loop
                for step in range(inference_time_s * fps):
                    # Check if stop was requested
                    if robot_controller and robot_controller.stop_execution:
                        print("🛑 Stop command received! Halting grasp and drop execution...")
                        break
                        
                    print(f'Iteration Start Time: {datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3]}')
                    start_time = time.perf_counter()
                    observation = robot.capture_observation()
                    
                    # # Save images
                    # image = observation['observation.images.phone']
                    # np_image = np.array(image)
                    # np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(os.path.join(output_dir, f"image_phone_{step}.jpg"), np_image)
                    
                    # image = observation['observation.images.on_robot']
                    # np_image = np.array(image)
                    # np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(os.path.join(output_dir, f"image_on_robot_{step}.jpg"), np_image)
                                        # Save images
                    for key in observation:
                        if 'images' in key:
                            image = observation[key]
                            np_image = np.array(image)
                            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(output_dir, f"image_{key}_{step}.jpg"), np_image)
                    
                    print(f"Step {step}")
                    
                    # Process observation
                    for name in observation:
                        observation[name] = observation[name].numpy()
                        print(name, observation[name].shape)

                    # Add task
                    observation["task"] = [task]
                        
                    print(f'Done processing observation: {datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3]}')
                    
                    try:
                        # Get action directly from client
                        action = await client.select_action(observation)
                        print(f'Get Action Time: {datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3]}')
                        action = torch.from_numpy(action)
                        action = action.squeeze(0)
                        print(f'Action Conversion Time: {datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3]}')
                        robot.send_action(action)
                        print(f'Robot Send Action Time: {datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3]}')
                        
                        # Calculate iteration performance
                        iteration_time = time.perf_counter() - start_time
                        iteration_ms = iteration_time * 1000
                        
                        # Update running averages
                        successful_steps += 1
                        running_total_time += iteration_time
                        iteration_times.append(iteration_time)
                        
                        # Calculate running averages
                        running_avg_ms = (running_total_time / successful_steps) * 1000
                        running_avg_fps = 1.0 / (running_total_time / successful_steps)
                        
                        # Calculate overall performance since start
                        elapsed_overall = time.perf_counter() - start_overall
                        overall_fps = successful_steps / elapsed_overall
                        
                        # Print performance stats
                        print(f"📊 Step {step}: {iteration_ms:.1f}ms | "
                              f"Avg: {running_avg_ms:.1f}ms ({running_avg_fps:.1f} FPS) | "
                              f"Overall: {overall_fps:.1f} FPS | "
                              f"Success: {successful_steps}/{step+1}")
                        
                    except Exception as e:
                        print(f"Failed to get action at step {step}: {e}")
                        # Print failure stats
                        elapsed_overall = time.perf_counter() - start_overall
                        overall_fps = successful_steps / elapsed_overall if successful_steps > 0 else 0
                        print(f"❌ Step {step}: FAILED | "
                              f"Overall: {overall_fps:.1f} FPS | "
                              f"Success: {successful_steps}/{step+1}")
                        continue

                    dt_s = time.perf_counter() - start_time
                    busy_wait(1 / fps - dt_s)

            finally:
                # Print final performance summary
                total_elapsed = time.perf_counter() - start_overall
                
                print("\n" + "="*60)
                print("📈 FINAL PERFORMANCE SUMMARY")
                print("="*60)
                
                if successful_steps > 0:
                    final_avg_ms = (running_total_time / successful_steps) * 1000
                    final_avg_fps = 1.0 / (running_total_time / successful_steps)
                    overall_fps = successful_steps / total_elapsed
                    success_rate = (successful_steps / (inference_time_s * fps)) * 100
                    
                    print(f"Total steps attempted: {inference_time_s * fps}")
                    print(f"Successful steps: {successful_steps}")
                    print(f"Success rate: {success_rate:.1f}%")
                    print(f"Average iteration time: {final_avg_ms:.1f}ms")
                    print(f"Average processing FPS: {final_avg_fps:.1f}")
                    print(f"Overall throughput FPS: {overall_fps:.1f}")
                    print(f"Total runtime: {total_elapsed:.1f}s")
                else:
                    print("❌ No successful iterations completed")
                
                print("="*60)
        
        # Check if execution was stopped early
        was_stopped = robot_controller and robot_controller.stop_execution
        if was_stopped:
            robot_controller.stop_execution = False  # Reset the flag
        
        return {
            "status": "success" if not was_stopped else "stopped",
            "message": f"Grasp and drop task {'stopped by user' if was_stopped else 'completed'}: {task}",
            "task": task,
            "successful_steps": successful_steps,
            "total_steps": inference_time_s * fps,
            "success_rate": f"{(successful_steps / (inference_time_s * fps)) * 100:.1f}%" if successful_steps > 0 else "0%",
            "was_stopped": was_stopped
        }
        
    except Exception as e:
        error_msg = f"Failed to complete grasp and drop task: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "task": task
        }
    finally:
        # Ensure robot cleanup happens regardless of success/failure
        if robot:
            try:
                robot.disconnect()
                print("✅ Robot disconnected in cleanup")
            except Exception as e:
                print(f"⚠️ Error during robot cleanup: {e}")


def stop_execution():
    """
    Stop the current grasp and drop execution
    
    Returns:
        dict: Status of the stop command
    """
    global robot_controller
    
    if robot_controller:
        robot_controller.stop_execution = True
        return {
            "status": "success",
            "message": "Stop command sent. The robot will halt its current task."
        }
    else:
        return {
            "status": "error", 
            "message": "No robot controller available to stop."
        }


def grasp_and_drop_sync(object_to_grasp, drop_location):
    """
    Synchronous wrapper for grasp_and_drop function
    """
    global robot_controller
    
    try:
        # Reset stop flag before starting
        if robot_controller:
            robot_controller.stop_execution = False
            print(f"🔄 Starting new grasp and drop task...")
        
        # Check if we're already in an async context
        try:
            # Try to get the current event loop
            current_loop = asyncio.get_running_loop()
            print("🔄 Running in existing event loop")
            
            # Create a new task in the current loop
            import concurrent.futures
            import functools
            
            def run_in_thread():
                # Create a new event loop in a separate thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(grasp_and_drop(object_to_grasp, drop_location))
                    return result
                finally:
                    new_loop.close()
            
            # Run in a separate thread to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                result = future.result(timeout=120)  # 2 minute timeout
                return result
                
        except RuntimeError:
            # No event loop running, we can create our own
            print("🔄 Creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(grasp_and_drop(object_to_grasp, drop_location))
                return result
            finally:
                loop.close()
                
    except Exception as e:
        print(f"❌ Error in grasp_and_drop_sync: {e}")
        return {
            "status": "error",
            "message": f"Failed to execute grasp and drop: {str(e)}"
        }


# OpenAI function schema
functions = [
    {
        "type": "function",
        "function": {
            "name": "go_to_position",
            "description": "Move the robot to a specific predefined position",
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "enum": ["rest", "zero", "rotated"],
                        "description": "The target position to move to"
                    }
                },
                "required": ["position"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grasp_and_drop_sync",
            "description": "Grasp an object and drop it at a specified location using AI inference",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_to_grasp": {
                        "type": "string",
                        "description": "The object to grasp (e.g., 'yellow cuboid', 'red ball', 'blue cup')"
                    },
                    "drop_location": {
                        "type": "string", 
                        "description": "Where to drop the object (e.g., 'bin', 'table', 'blue container')"
                    }
                },
                "required": ["object_to_grasp", "drop_location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_execution",
            "description": "Stop the current robot task execution immediately",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def handle_function_call(function_name, arguments):
    """Handle the function call based on the function name"""
    if function_name == "go_to_position":
        return go_to_position(**arguments)
    elif function_name == "grasp_and_drop_sync":
        return grasp_and_drop_sync(**arguments)
    elif function_name == "stop_execution":
        return stop_execution()
    else:
        return {"status": "error", "message": f"Unknown function: {function_name}"}

class VoiceControlledRobot:
    def __init__(self):
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        self.connection = None
        self.microphone = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.command_queue = queue.Queue()  # Queue for voice commands
        self.stop_execution = False  # Flag to stop grasp and drop execution
        
        # Audio settings
        self.RATE = 16000
        self.CHUNK = 8000
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
    def setup_microphone(self):
        """Initialize PyAudio for microphone input"""
        self.microphone = pyaudio.PyAudio()
        
    def audio_callback(self, input_data, frame_count, time_info, status_flag):
        """Callback function for audio stream"""
        if self.is_running:
            try:
                self.audio_queue.put_nowait(input_data)
            except queue.Full:
                pass
        return (input_data, pyaudio.paContinue)
    
    def process_voice_commands(self, message):
        """Process voice commands using OpenAI function calling"""
        if not message.strip():
            return
            
        print(f"🎤 Processing command: '{message}'")
        
        # Reset stop flag for new commands (except for explicit stop commands)
        if "stop" not in message.lower():
            self.stop_execution = False
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that controls a robot. You have access to functions that can: 1) Move the robot to predefined positions (rest, zero, rotated), 2) Perform grasp and drop tasks where you grasp an object and drop it at a location, and 3) Stop the current robot execution immediately. When users ask to move objects, pick things up, or put things somewhere, use the grasp_and_drop_sync function. For simple positioning, use go_to_position. When users say 'stop', 'halt', 'abort', or similar commands, use stop_execution."},
            {"role": "user", "content": message}
        ]
        
        try:
            # Make the initial API call
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=functions,
                tool_choice="auto"
            )
            
            message_response = response.choices[0].message
            messages.append(message_response)
            
            # Check if the assistant wants to call a function
            if message_response.tool_calls:
                print("🧠 AI detected a robot command!")
                
                # Handle each function call
                for tool_call in message_response.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"🎯 Executing: {function_name} with arguments: {arguments}")
                    
                    # Execute the function
                    function_result = handle_function_call(function_name, arguments)
                    
                    # Add the function result to the conversation
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": json.dumps(function_result)
                    })
                
                # Get the final response from the assistant
                final_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                print(f"🤖 Assistant: {final_response.choices[0].message.content}")
            else:
                print(f"🤖 Assistant: {message_response.content}")
                
        except Exception as e:
            print(f"❌ Error processing command: {e}")
        
    async def start_transcription(self):
        """Start the live transcription process"""
        try:
            # Create websocket connection
            self.connection = self.deepgram.listen.websocket.v("1")
            
            # Set up event handlers - capture reference to our robot instance
            robot_instance = self
            
            def on_message(ws_self, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                if sentence.strip():
                    print(f"📝 Transcribed: {sentence}")
                    # Process the command in a separate thread to avoid blocking
                    threading.Thread(
                        target=robot_instance.process_voice_commands, 
                        args=(sentence,), 
                        daemon=True
                    ).start()
                    
            def on_error(ws_self, error, **kwargs):
                print(f"❌ Transcription Error: {error}")
                
            def on_close(ws_self, close_event, **kwargs):
                print("🔌 Connection closed")
            
            # Register event handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.connection.on(LiveTranscriptionEvents.Error, on_error)
            self.connection.on(LiveTranscriptionEvents.Close, on_close)
            
            # Configure transcription options
            options = LiveOptions(
                model="nova-3",
                language="en-US",
                encoding="linear16",
                sample_rate=self.RATE,
                channels=self.channels,
                interim_results=False,
                punctuate=True,
                smart_format=True
            )
            
            # Start the connection
            success = self.connection.start(options)
            if success:
                print("🎤 Connected to Deepgram! Start speaking...")
                print("💬 Voice command examples:")
                print("   Position commands:")
                print("     • 'Move to the zero position'")
                print("     • 'Go to rest position'")
                print("   Grasp and drop commands:")
                print("     • 'Pick up the yellow cuboid and put it in the bin'")
                print("     • 'Grasp the red ball and drop it on the table'")
                print("     • 'Move the blue cup to the container'")
                print("   Stop commands:")
                print("     • 'Stop'")
                print("     • 'Halt the robot'")
                print("     • 'Abort the current task'")
                self.is_running = True
                
                # Start audio capture
                await self.capture_audio()
            else:
                print("❌ Failed to connect to Deepgram")
                
        except Exception as e:
            print(f"❌ Error in transcription: {e}")
            
    async def capture_audio(self):
        """Capture audio from microphone and send to Deepgram"""
        # Start audio stream
        stream = self.microphone.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        stream.start_stream()
        
        try:
            while self.is_running and stream.is_active():
                try:
                    # Get audio data from queue (non-blocking)
                    audio_data = self.audio_queue.get(timeout=0.1)
                    if self.connection:
                        self.connection.send(audio_data)
                except queue.Empty:
                    # No audio data available, continue
                    await asyncio.sleep(0.01)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping voice control...")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice control...")
        finally:
            self.is_running = False
            stream.stop_stream()
            stream.close()
            if self.connection:
                self.connection.finish()
                
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.microphone:
            self.microphone.terminate()

def test_api_key():
    """Test if your Deepgram API key works"""
    try:
        client = DeepgramClient(DEEPGRAM_API_KEY)
        response = client.listen.rest.v("1").transcribe_url(
            {"url": "https://dpgr.am/spacewalk.wav"},
            {"model": "nova-3"}
        )
        
        transcript = response.results.channels[0].alternatives[0].transcript
        print("✅ Deepgram API key is valid!")
        return True
    except Exception as e:
        print(f"❌ Deepgram API key test failed: {e}")
        return False

def test_openai_key():
    """Test if OpenAI API key works"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ OpenAI API key is valid!")
        return True
    except Exception as e:
        print(f"❌ OpenAI API key test failed: {e}")
        return False

async def main():
    """Main function to run the voice-controlled robot"""
    global robot_controller
    
    print("🤖 === Voice-Controlled Robot System ===")
    print("🔍 Testing API keys...")
    
    if not test_api_key():
        print("❌ Please check your DEEPGRAM_API_KEY")
        return
        
    if not test_openai_key():
        print("❌ Please check your OPENAI_API_KEY")
        return
    
    print("✅ All API keys valid!")
    print("\n🎯 Available robot capabilities:")
    print("   📍 Position Control:")
    print("     • 'rest' - Home/rest position")
    print("     • 'zero' - Zero/origin position") 
    print("     • 'rotated' - Rotated position")
    print("   🤲 Grasp and Drop:")
    print("     • Pick up objects and place them at locations")
    print("     • Uses AI inference for complex manipulation")
    print("   🛑 Stop Control:")
    print("     • Say 'stop' to halt current execution")
    print("\n💡 Voice command examples:")
    print("   📍 Position commands:")
    print("     • 'Move to the zero position'")
    print("     • 'Go to rest position'")
    print("   🤲 Grasp and drop commands:")
    print("     • 'Pick up the yellow cuboid and put it in the bin'")
    print("     • 'Grasp the red ball and drop it on the table'")
    print("     • 'Move the blue cup to the container'")
    print("   🛑 Stop commands:")
    print("     • 'Stop'")
    print("     • 'Halt the robot'")
    print("     • 'Abort the current task'")
    print("\n🎤 Starting voice control...")
    print("   Press Ctrl+C to stop\n")
    
    # Start voice-controlled robot
    robot = VoiceControlledRobot()
    robot_controller = robot  # Set global reference for stop functionality
    robot.setup_microphone()
    
    try:
        await robot.start_transcription()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down voice-controlled robot...")
    finally:
        robot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())